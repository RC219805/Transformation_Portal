"""Adversarial coverage for private, independently paired candidate evidence."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from transformation_portal.lux_depth_v4 import evaluation as ev

pytestmark = pytest.mark.unit


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "source"
    images = source / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (4, 3), (13, 45, 79)).save(images / "scene.png")
    Image.new("RGB", (4, 3), (13, 45, 79)).save(images / "scene.tif")
    (images / "duplicate.png").write_bytes((images / "scene.png").read_bytes())
    target = tmp_path / "private" / "corpus.json"
    ev.freeze_corpus(source, ["images"], target, "a" * 40)
    return target


@pytest.fixture
def spec(corpus):
    identity = {key: "a" * 64 for key in ev.IDENTITY_DIGESTS}
    identity["implementation_commit"] = "a" * 40
    candidate = {**identity, "source_sha256": "b" * 64, "plan_sha256": "b" * 64, "runtime_sha256": "b" * 64}
    return {
        "schema": ev.SPEC_SCHEMA,
        "corpus_sha256": ev.file_sha256(corpus),
        "repeats": 20,
        "scenarios": ["cold"],
        "experiment": "infrastructure",
        "variants": {"v3": {"identity": identity}, "v4": {"identity": candidate}},
    }


def _runner(request):
    manifest = json.loads(request.corpus_path.read_text())
    hashes = sorted({row["sha256"] for row in manifest["files"]})
    artifacts = []
    for index, digest in enumerate(hashes):
        image = request.output_dir / f"image_{index}.png"
        Image.new("RGB", (4, 3), (13, 45, 79)).save(image)
        artifacts.append({"path": image.name, "sha256": ev.file_sha256(image), "kind": "image", "input_sha256": digest})
    evidence = request.output_dir / "execution.json"
    evidence.write_text(json.dumps({"schema": "test.execution.evidence"}))
    artifacts.append({"path": evidence.name, "sha256": ev.file_sha256(evidence), "kind": "evidence"})
    return {
        "schema": ev.RECEIPT_SCHEMA,
        "complete": True,
        "synthetic": False,
        "executed_backend": "test_instrumentation",
        "corpus_sha256": request.corpus_sha256,
        "identity": dict(request.identity),
        "host": dict(request.host),
        "input_sha256s": hashes,
        "artifacts": artifacts,
        "warmup_complete": True,
        "session_id": request.variant,
        "cache_authorized": True,
        "cache_state": request.scenario.removeprefix("cache_"),
    }


def _run(tmp_path, corpus, spec, monkeypatch, runner=_runner, delta=0.08):
    values = []
    elapsed = 0.0
    for _, _, variant in ev._sequence(spec):
        values.append(elapsed)
        elapsed += 1.0 if variant == "v3" else 1.0 + delta
        values.append(elapsed)
        elapsed += 0.25
    ticks = iter(values)
    monkeypatch.setattr(ev.time, "perf_counter", lambda: next(ticks))
    return ev.run_evaluation(corpus, spec, tmp_path / "observations", runner)


def test_freeze_deduplicates_bytes_and_links_candidate_scene_stems(corpus):
    manifest = ev.verify_corpus(corpus)
    rows = {row["path"]: row for row in manifest["files"]}
    assert len(rows) == 3
    assert sum(row["duplicate_of"] is None for row in rows.values()) == 2
    assert rows["images/scene.png"]["scene_id"] == rows["images/scene.tif"]["scene_id"]
    assert rows["images/scene.png"]["metadata"]["width"] == 4
    assert corpus.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        ev.freeze_corpus(Path(manifest["source_root"]), ["images"], corpus, "a" * 40)


def test_frozen_input_changes_fail_closed(corpus):
    manifest = ev.verify_corpus(corpus)
    source = Path(manifest["source_root"]) / manifest["files"][0]["path"]
    source.write_bytes(b"changed")
    with pytest.raises(ev.EvaluationError, match="Frozen input changed"):
        ev.verify_corpus(corpus)


def test_manifest_changes_fail_closed(corpus):
    digest = ev.file_sha256(corpus)
    corpus.write_text(corpus.read_text() + "\n")
    with pytest.raises(ev.EvaluationError, match="manifest changed"):
        ev.verify_corpus(corpus, digest)


def test_private_outputs_cannot_be_committed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(ev.EvaluationError, match="outside a Git worktree"):
        ev._write_private(repo / "private.json", {"secret": "fixture"})


def test_corpus_root_traversal_is_rejected(tmp_path):
    with pytest.raises(ev.EvaluationError, match="relative paths"):
        ev.freeze_corpus(tmp_path, ["../elsewhere"], tmp_path / "out.json", "a" * 40)


def test_symlink_input_escape_is_rejected(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = tmp_path / "elsewhere.png"
    Image.new("RGB", (2, 2)).save(source)
    (inputs / "alias.png").symlink_to(source)
    with pytest.raises(ev.EvaluationError, match="not symlinks"):
        ev.freeze_corpus(tmp_path, ["inputs"], tmp_path / "out.json", "a" * 40)


@pytest.mark.parametrize("count", [0, 3, 19, True, 20.0])
def test_insufficient_or_invalid_repeat_count_is_rejected(spec, count):
    spec["repeats"] = count
    with pytest.raises(ev.EvaluationError, match="independent pairs"):
        ev.validate_spec(spec)


@pytest.mark.parametrize("field", ["model_sha256", "processing_sha256", "dependency_sha256", "interpreter_sha256"])
def test_infrastructure_cannot_mix_algorithm_or_environment_changes(spec, field):
    spec["variants"]["v4"]["identity"][field] = "c" * 64
    with pytest.raises(ev.EvaluationError, match="differs"):
        ev.validate_spec(spec)


@pytest.mark.parametrize("experiment,field", [("model", "model_sha256"), ("photography", "processing_sha256")])
def test_explicit_experiment_may_change_only_its_selected_identity(spec, experiment, field):
    spec["experiment"] = experiment
    spec["variants"]["v4"]["identity"][field] = "c" * 64
    ev.validate_spec(spec)


def test_paired_batches_are_independent_and_photographic_review_never_autopasses(tmp_path, corpus, spec, monkeypatch):
    run = _run(tmp_path, corpus, spec, monkeypatch)
    observations = json.loads(run.read_text())["observations"]
    assert [row["variant"] for row in observations[:4]] == ["v3", "v4", "v4", "v3"]
    result = ev.compare_run(run)
    cold = result["scenarios"]["cold"]
    assert cold["v3"]["independent_batches"] == 20
    assert cold["v4"]["independent_batches"] == 20
    assert cold["p95_delta_percent"] == pytest.approx(8)
    assert cold["performance_verdict"] == "pass"
    assert result["photographic_review"] == "required"
    assert result["candidate_acceptance"] == "not_established"
    assert result["automatic_enforcement_eligible"] is False


@pytest.mark.parametrize("delta,verdict", [(0.12, "warn"), (0.16, "fail")])
def test_existing_apex_relative_thresholds_are_reported(tmp_path, corpus, spec, monkeypatch, delta, verdict):
    run = _run(tmp_path, corpus, spec, monkeypatch, delta=delta)
    assert ev.compare_run(run)["scenarios"]["cold"]["performance_verdict"] == verdict


def test_uncertainty_resamples_batch_pairs_together_and_is_reproducible():
    baseline = [float(index + 1) for index in range(20)]
    samples = {"v3": baseline, "v4": [value * 1.08 for value in baseline]}
    first = ev._paired_uncertainty(samples, seed="a" * 64)
    assert first == ev._paired_uncertainty(samples, seed="a" * 64)
    # Independent variant resampling would spuriously widen this perfectly
    # paired proportional example; image counts never enter the calculation.
    assert first["p95_delta_percent_interval"] == pytest.approx([8.0, 8.0])
    assert first["independent_pairs"] == 20
    assert first["production_confidence_gate"] == "not_evaluated"


def test_uncertainty_exposes_variable_pair_outcomes():
    samples = {"v3": [10.0] * 20, "v4": [8.0] * 19 + [20.0]}
    interval = ev._paired_uncertainty(samples, seed="b" * 64)["p95_delta_percent_interval"]
    assert interval[0] < 0 < interval[1]


def test_uncertainty_rejects_incomplete_pairs():
    with pytest.raises(ev.EvaluationError, match="complete independent batch pairs"):
        ev._paired_uncertainty({"v3": [1.0] * 20, "v4": [1.0] * 19}, seed="a" * 64)


@pytest.mark.parametrize("mutation", ["synthetic", "incomplete", "identity", "host", "inputs", "traversal", "missing_image"])
def test_false_or_incomplete_receipts_are_rejected(tmp_path, corpus, spec, monkeypatch, mutation):
    def invalid(request):
        receipt = _runner(request)
        if mutation == "synthetic":
            receipt["synthetic"] = True
        elif mutation == "incomplete":
            receipt["complete"] = False
        elif mutation == "identity":
            receipt["identity"]["model_sha256"] = "f" * 64
        elif mutation == "host":
            receipt["host"]["node"] = "different-machine"
        elif mutation == "inputs":
            receipt["input_sha256s"].pop()
        elif mutation == "traversal":
            receipt["artifacts"][0]["path"] = "../image.png"
        else:
            receipt["artifacts"].pop(0)
        return receipt

    with pytest.raises(ev.EvaluationError):
        _run(tmp_path, corpus, spec, monkeypatch, runner=invalid)
    assert not (tmp_path / "observations" / "run.json").exists()


def test_callback_cannot_mutate_expected_identity(tmp_path, corpus, spec, monkeypatch):
    def runner(request):
        with pytest.raises(TypeError):
            request.identity["model_sha256"] = "b" * 64
        return _runner(request)

    _run(tmp_path, corpus, spec, monkeypatch, runner=runner)


@pytest.mark.parametrize("mutation", ["artifact", "duplicate", "missing", "spec", "summary"])
def test_comparison_rejects_changed_or_duplicated_evidence(tmp_path, corpus, spec, monkeypatch, mutation):
    run = _run(tmp_path, corpus, spec, monkeypatch)
    payload = json.loads(run.read_text())
    if mutation == "artifact":
        (run.parent / "cold/pair_0000/v3/image_0.png").write_bytes(b"tampered")
    elif mutation == "duplicate":
        payload["observations"][1] = copy.deepcopy(payload["observations"][0])
    elif mutation == "missing":
        payload["observations"].pop()
    elif mutation == "spec":
        (run.parent / "spec.json").write_text("{}")
    else:
        payload["observations"][0]["elapsed_seconds"] = 12
    run.write_text(json.dumps(payload))
    with pytest.raises(ev.EvaluationError):
        ev.compare_run(run)


def test_stateful_warm_callback_is_supported(tmp_path, corpus, spec, monkeypatch):
    spec["scenarios"] = ["warm"]
    run = _run(tmp_path, corpus, spec, monkeypatch)
    assert ev.compare_run(run)["scenarios"]["warm"]["v3"]["independent_batches"] == 20


def test_warm_reinitialization_is_not_reported_as_warm(tmp_path, corpus, spec, monkeypatch):
    spec["scenarios"] = ["warm"]

    def runner(request):
        receipt = _runner(request)
        receipt["session_id"] = f"{request.variant}-{request.pair_index}"
        return receipt

    with pytest.raises(ev.EvaluationError, match="session changed"):
        _run(tmp_path, corpus, spec, monkeypatch, runner=runner)


def test_command_adapter_cannot_fake_stateful_scenarios(spec):
    spec["scenarios"] = ["warm"]
    with pytest.raises(ev.EvaluationError, match="stateful Python callback"):
        ev.command_runner(spec)


@pytest.mark.parametrize("scenario", ["cache_hit", "cache_miss"])
def test_cache_state_requires_matching_authorization(tmp_path, corpus, spec, monkeypatch, scenario):
    spec["scenarios"] = [scenario]

    def invalid(request):
        receipt = _runner(request)
        receipt["cache_authorized"] = False
        return receipt

    with pytest.raises(ev.EvaluationError, match="governed cache evidence"):
        _run(tmp_path, corpus, spec, monkeypatch, runner=invalid)


def test_command_adapter_passes_receipt_path_without_shell(tmp_path, corpus, spec):
    code = "import os,pathlib;pathlib.Path(os.environ['TP_LUX_EVAL_RECEIPT']).write_text('{}')"
    for variant in spec["variants"].values():
        variant["command"] = [sys.executable, "-c", code]
    runner = ev.command_runner(spec)
    request = ev._request(spec, corpus, tmp_path, "cold", 0, "v3")
    request.output_dir.mkdir(parents=True)
    assert runner(request) == {}
    assert (request.output_dir / "stdout.log").is_file()


def test_command_timeout_cannot_complete_evidence(tmp_path, corpus, spec):
    spec["timeout_seconds"] = 0.05
    for variant in spec["variants"].values():
        variant["command"] = [sys.executable, "-c", "import time; time.sleep(30)"]
    runner = ev.command_runner(spec)
    request = ev._request(spec, corpus, tmp_path, "cold", 0, "v3")
    request.output_dir.mkdir(parents=True)
    with pytest.raises(ev.EvaluationError, match="command failed"):
        runner(request)


@pytest.mark.parametrize("invalid", [[], None, {"schema": ev.SPEC_SCHEMA, "corpus_sha256": "0" * 64}])
def test_invalid_specs_have_explicit_fail_closed_errors(invalid):
    with pytest.raises(ev.EvaluationError):
        ev.validate_spec(invalid)
