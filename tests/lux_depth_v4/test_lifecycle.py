"""Preparation remains read-only and preserves explicit physical/content boundaries."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError
from transformation_portal.lux_depth_v4 import lifecycle
from transformation_portal.lux_depth_v4.io import directory_path, snapshot, write_evidence
from transformation_portal.lux_depth_v4.lifecycle import LuxDepthV4Request, prepare

pytestmark = pytest.mark.unit


@pytest.fixture
def v4_request(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    Image.new("RGB", (4, 3), (17, 43, 79)).save(source / "photograph.png")
    return LuxDepthV4Request(source, tmp_path / "output", input_color="srgb", cache_dir=tmp_path / "cache")


def test_prepare_is_read_only_and_does_not_start_a_model_worker(v4_request, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU preparation must not spawn a worker or load a model")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    prepared = prepare(v4_request)
    assert not v4_request.output_dir.exists()
    assert not v4_request.cache_dir.exists()
    payload = prepared.plan.to_payload()
    assert payload["schema"] == "tp.execution.plan.v2"
    assert payload["model"]["canonical_key"] == "da3_metric"
    assert payload["inputs"][0]["sha256"] == hashlib.sha256((v4_request.input_dir / "photograph.png").read_bytes()).hexdigest()
    assert str(v4_request.output_dir) not in prepared.canonical_plan_bytes.decode()
    with pytest.raises(FrozenInstanceError):
        prepared.runtime_python = "different"


def test_plan_is_deterministic_and_independent_of_publication_directory(v4_request):
    first = prepare(v4_request)
    second = prepare(replace(v4_request, output_dir=v4_request.output_dir.with_name("second")))
    assert first.canonical_plan_bytes == second.canonical_plan_bytes
    assert first.output_root != second.output_root
    changed = prepare(replace(v4_request, strength=0.75))
    assert first.plan.plan_fingerprint_sha256 != changed.plan.plan_fingerprint_sha256


def test_selection_is_sorted_and_content_changes_change_plan(v4_request):
    nested = v4_request.input_dir / "nested"
    nested.mkdir()
    Image.new("RGB", (4, 3)).save(nested / "second.png")
    first = prepare(v4_request)
    assert [item["path"] for item in first.plan.to_payload()["inputs"]] == ["nested/second.png", "photograph.png"]
    Image.new("RGB", (4, 3), (2, 4, 6)).save(nested / "second.png")
    assert prepare(v4_request).canonical_plan_bytes != first.canonical_plan_bytes


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_size", 15),
        ("target_size", True),
        ("input_color", "display_p3"),
        ("strength", float("nan")),
        ("strength", float("inf")),
        ("strength", True),
        ("clarity", -1),
        ("preview_maps", "false"),
        ("max_pixels", 0),
        ("max_input_bytes", -1),
        ("max_output_bytes", 2**50),
        ("wall_time_seconds", 0),
        ("memory_mib", 1),
        ("device", "cuda"),
    ],
)
def test_invalid_request_fails_before_input_hashing_or_device_probe(v4_request, monkeypatch, field, value):
    monkeypatch.setattr(
        lifecycle, "snapshot", lambda *args, **kwargs: pytest.fail("invalid v4_request read photographic bytes")
    )
    with pytest.raises(ValueError):
        prepare(replace(v4_request, **{field: value}))
    assert not v4_request.output_dir.exists()


@pytest.mark.parametrize("placement", ["same", "descendant", "ancestor", "cache_input", "cache_output", "cache_ancestor"])
def test_input_output_cache_roots_must_be_disjoint(v4_request, placement):
    if placement == "same":
        changed = replace(v4_request, output_dir=v4_request.input_dir)
    elif placement == "descendant":
        changed = replace(v4_request, output_dir=v4_request.input_dir / "results")
    elif placement == "ancestor":
        changed = replace(v4_request, output_dir=v4_request.input_dir.parent)
    elif placement == "cache_input":
        changed = replace(v4_request, cache_dir=v4_request.input_dir / "cache")
    elif placement == "cache_output":
        changed = replace(v4_request, cache_dir=v4_request.output_dir / "cache")
    else:
        changed = replace(v4_request, cache_dir=v4_request.input_dir.parent)
    with pytest.raises(ValueError, match="separate"):
        prepare(changed)
    assert not v4_request.output_dir.exists()


@pytest.mark.parametrize("kind", ["input", "input_ancestor", "output", "output_ancestor", "cache"])
def test_linked_roots_and_existing_ancestors_are_rejected(v4_request, tmp_path, kind):
    alias = tmp_path / "alias"
    if kind == "input":
        alias.symlink_to(v4_request.input_dir, target_is_directory=True)
        changed = replace(v4_request, input_dir=alias)
    elif kind == "input_ancestor":
        alias.symlink_to(tmp_path, target_is_directory=True)
        changed = replace(v4_request, input_dir=alias / "input")
    elif kind == "output":
        alias.symlink_to(v4_request.input_dir, target_is_directory=True)
        changed = replace(v4_request, output_dir=alias)
    elif kind == "output_ancestor":
        alias.symlink_to(tmp_path, target_is_directory=True)
        changed = replace(v4_request, output_dir=alias / "new-output")
    else:
        alias.symlink_to(tmp_path, target_is_directory=True)
        changed = replace(v4_request, cache_dir=alias / "new-cache")
    with pytest.raises((ValueError, ArtifactEvidenceError)):
        prepare(changed)


def test_linked_input_subtree_is_not_silently_omitted(v4_request, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    Image.new("RGB", (2, 2)).save(other / "hidden.png")
    (v4_request.input_dir / "linked").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="linked directories"):
        prepare(v4_request)


def test_symlink_image_and_hardlink_image_are_rejected(v4_request):
    source = v4_request.input_dir / "photograph.png"
    alias = v4_request.input_dir / "alias.png"
    alias.symlink_to(source)
    with pytest.raises(ArtifactEvidenceError):
        prepare(v4_request)
    alias.unlink()
    os.link(source, alias)
    with pytest.raises(ValueError, match="link aliases"):
        prepare(v4_request)


def test_missing_and_empty_selection_fail_before_device_probe(tmp_path, monkeypatch):
    import transformation_portal.lux_depth_v4.backend as backend

    monkeypatch.setattr(backend, "probe_device", lambda *_args: pytest.fail("empty input probed a runtime"))
    source = tmp_path / "empty"
    with pytest.raises(ValueError, match="existing"):
        prepare(LuxDepthV4Request(source, tmp_path / "output", device="auto"))
    source.mkdir()
    with pytest.raises(ValueError, match="no supported"):
        prepare(LuxDepthV4Request(source, tmp_path / "output", device="auto"))


def test_auto_device_is_resolved_once_without_starting_backend_session(v4_request, monkeypatch):
    import transformation_portal.lux_depth_v4.backend as backend

    probes = []
    monkeypatch.setattr(backend, "probe_device", lambda interpreter, device: probes.append((interpreter, device)) or "mps")
    monkeypatch.setattr(backend, "DA3Session", lambda *_args, **_kwargs: pytest.fail("prepare started inference"))
    prepared = prepare(replace(v4_request, device="auto", runtime_python=".runtime/venv/bin/python"))
    assert len(probes) == 1
    assert probes[0][1] == "auto"
    assert prepared.plan.to_payload()["device"] == "mps"
    assert not v4_request.output_dir.exists()


def test_venv_interpreter_symlink_spelling_is_preserved(v4_request, tmp_path):
    base = tmp_path / "base-python"
    base.write_bytes(b"fixture")
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(base)
    prepared = prepare(replace(v4_request, runtime_python=str(interpreter), raw_python=str(interpreter)))
    assert prepared.runtime_python == str(interpreter)
    assert prepared.raw_python == str(interpreter)


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_snapshot_rejects_invalid_byte_budgets(v4_request, maximum):
    with pytest.raises(ValueError, match="positive integer"):
        snapshot(v4_request.input_dir, v4_request.input_dir / "photograph.png", maximum_bytes=maximum)


def test_snapshot_without_retention_still_hashes_all_bytes(v4_request):
    path = v4_request.input_dir / "photograph.png"
    data, record = snapshot(v4_request.input_dir, path, maximum_bytes=1024, retain_bytes=False)
    assert data == b""
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["size_bytes"] == path.stat().st_size
    with pytest.raises(ValueError, match="bounded regular"):
        snapshot(v4_request.input_dir, path, maximum_bytes=1)


def test_same_size_mutation_during_snapshot_is_rejected(v4_request, monkeypatch):
    path = v4_request.input_dir / "photograph.png"
    original = os.read
    changed = False

    def changing(descriptor, count):
        nonlocal changed
        data = original(descriptor, count)
        if data and not changed:
            changed = True
            path.write_bytes(b"x" * path.stat().st_size)
        return data

    monkeypatch.setattr(os, "read", changing)
    with pytest.raises(ArtifactEvidenceError, match="captured inode"):
        snapshot(v4_request.input_dir, path, maximum_bytes=1024)


def test_root_replacement_during_preparation_is_rejected(v4_request, monkeypatch):
    original = lifecycle.snapshot
    changed = False

    def replacing(root, path, **kwargs):
        nonlocal changed
        result = original(root, path, **kwargs)
        if not changed:
            changed = True
            root.rename(root.with_name("moved-original"))
            root.mkdir()
        return result

    monkeypatch.setattr(lifecycle, "snapshot", replacing)
    with pytest.raises(ArtifactEvidenceError, match="pinned directory"):
        prepare(v4_request)
    assert not v4_request.output_dir.exists()


def test_evidence_writer_cannot_escape_or_follow_a_symlink(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"untouched")
    (output / "alias.json").symlink_to(outside)
    for name in ("../outside.json", "alias.json"):
        with pytest.raises(ArtifactEvidenceError):
            write_evidence(output, name, b"changed")
    assert outside.read_bytes() == b"untouched"
    write_evidence(output, "complete.json", b"{}")
    assert (output / "complete.json").read_bytes() == b"{}"


def test_standard_macos_tmp_alias_remains_supported(v4_request):
    canonical = directory_path(v4_request.input_dir)
    assert canonical == v4_request.input_dir
    assert directory_path(v4_request.output_dir, allow_missing=True) == v4_request.output_dir
    assert not v4_request.output_dir.exists()


def test_replaced_prepared_binding_is_revalidated_without_model_loading(v4_request, monkeypatch):
    prepared = prepare(v4_request)
    monkeypatch.setattr(lifecycle, "resolve_model_contract", lambda *_args: pytest.fail("binding check resolved a model"))
    lifecycle.validate_prepared_bindings(prepared)
    with pytest.raises(ValueError, match="non-overlapping"):
        lifecycle.validate_prepared_bindings(replace(prepared, output_root=prepared.input_root))
    prepared.output_root.symlink_to(prepared.input_root, target_is_directory=True)
    with pytest.raises(ArtifactEvidenceError):
        lifecycle.validate_prepared_bindings(prepared)


def test_companion_namespace_must_remain_separate_from_cache_and_outputs(v4_request):
    from transformation_portal.ingest.canonical_json import canonicalize_json

    parent = v4_request.output_dir.parent
    source = v4_request.input_dir / "photograph.png"
    manifest = parent / "companions.json"
    manifest.write_bytes(
        canonicalize_json(
            {
                "schema": "tp.lux.companions.v1",
                "inputs": [
                    {
                        "path": "photograph.png",
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "calibration": {
                            "coordinate_space": "canonical_master",
                            "width": 4,
                            "height": 3,
                            "fx": 300,
                            "fy": 300,
                            "cx": 1.5,
                            "cy": 1,
                            "source": "measured fixture",
                        },
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="Companion root must be separate"):
        prepare(replace(v4_request, companions_manifest=manifest))
    assert not v4_request.output_dir.exists()


def test_companions_cli_plan_is_exact_and_read_only(v4_request, capfd):
    import json

    from transformation_portal.ingest.canonical_json import canonicalize_json
    from transformation_portal.lux_depth_v4.__main__ import main

    root = v4_request.output_dir.parent / "companions"
    root.mkdir()
    source = v4_request.input_dir / "photograph.png"
    manifest = root / "companions.json"
    manifest.write_bytes(
        canonicalize_json(
            {
                "schema": "tp.lux.companions.v1",
                "inputs": [
                    {
                        "path": "photograph.png",
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "calibration": {
                            "coordinate_space": "canonical_master",
                            "width": 4,
                            "height": 3,
                            "fx": 300,
                            "fy": 300,
                            "cx": 1.5,
                            "cy": 1,
                            "source": "measured fixture",
                        },
                    }
                ],
            }
        )
    )
    assert (
        main(
            [
                "--input-dir",
                str(v4_request.input_dir),
                "--output-dir",
                str(v4_request.output_dir),
                "--input-color",
                "srgb",
                "--companions-manifest",
                str(manifest),
                "--plan",
            ]
        )
        == 0
    )
    output = capfd.readouterr().out
    prepared = prepare(replace(v4_request, companions_manifest=manifest))
    assert output.encode() == prepared.canonical_plan_bytes
    payload = json.loads(output)
    assert payload["companions_manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert not v4_request.output_dir.exists()
