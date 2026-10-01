"""Regression coverage for the project duplicate resolver's read-only paths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/utilities/resolve_750_picacho_duplicates.py"


def _run_cli(base_dir, *args):
    return subprocess.run([sys.executable, str(SCRIPT), str(base_dir), *args], capture_output=True, text=True, check=False)


def test_missing_base_directory_exits_without_traceback(tmp_path):
    result = _run_cli(tmp_path / "missing")

    assert result.returncode == 1
    assert "Base directory not found" in result.stdout
    assert "Traceback" not in result.stderr


def test_tif_source_is_included_in_canonical_manifest(tmp_path):
    source_dir = tmp_path / "TIFFs" / "_TIFFs"
    source_dir.mkdir(parents=True)
    source = source_dir / "scene.tif"
    source.write_bytes(b"source fixture")

    result = _run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "canonical_sources_manifest.json").read_text())

    assert manifest["resolution_summary"]["canonical_sources_identified"] == 1
    assert manifest["canonical_sources"]["scene"]["path"] == str(source)


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("canonical_name", ["scene.tif", "scene-2.tif"])
def test_cleanup_retains_outputs_with_ambiguous_source_ownership(tmp_path, dry_run, canonical_name):
    source_dir = tmp_path / "TIFFs" / "_TIFFs"
    source_dir.mkdir(parents=True)
    canonical = source_dir / canonical_name
    canonical.write_bytes(b"canonical source")
    alternate_dir = tmp_path / "16-Bit_EXRs"
    alternate_dir.mkdir()
    alternate = alternate_dir / "scene.exr"
    alternate.write_bytes(b"alternate source")
    os.utime(alternate, (100, 100))
    os.utime(canonical, (200, 200))
    output_dir = tmp_path / "Maximum_Quality_Final"
    output_dir.mkdir()
    output = output_dir / f"{canonical.stem}_enhanced.tif"
    output.write_bytes(b"canonical result")
    result = _run_cli(tmp_path, *([] if dry_run else ["--cleanup"]))

    assert result.returncode == 0, result.stderr
    assert "Files to archive:" not in result.stdout
    assert output.read_bytes() == b"canonical result"
    assert not (output_dir / "_archived_duplicates").exists()


@pytest.mark.parametrize("dry_run", [True, False])
def test_cleanup_retains_another_scenes_canonical_output(tmp_path, dry_run):
    exr_dir = tmp_path / "16-Bit_EXRs"
    exr_dir.mkdir()
    tif_dir = tmp_path / "TIFFs" / "_TIFFs"
    tif_dir.mkdir(parents=True)
    for directory, name, modified in [
        (exr_dir, "scene.exr", 200),
        (tif_dir, "2-scene.tif", 100),
        (tif_dir, "2-scene-detail.tif", 200),
    ]:
        source = directory / name
        source.write_bytes(b"source fixture")
        os.utime(source, (modified, modified))
    output_dir = tmp_path / "Maximum_Quality_Final"
    output_dir.mkdir()
    output = output_dir / "2-scene-detail_enhanced.tif"
    output.write_bytes(b"another scene canonical result")

    result = _run_cli(tmp_path, *([] if dry_run else ["--cleanup"]))

    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "canonical_sources_manifest.json").read_text())
    assert Path(manifest["canonical_sources"]["scene-detail"]["path"]).name == "2-scene-detail.tif"
    assert "Files to archive:" not in result.stdout
    assert output.read_bytes() == b"another scene canonical result"
    assert not (output_dir / "_archived_duplicates").exists()


def test_cleanup_still_archives_unambiguously_alternate_outputs(tmp_path):
    source_dir = tmp_path / "16-Bit_EXRs"
    source_dir.mkdir()
    canonical = source_dir / "scene.exr"
    canonical.write_bytes(b"canonical source")
    alternate = source_dir / "2-scene.exr"
    alternate.write_bytes(b"alternate source")
    os.utime(alternate, (100, 100))
    os.utime(canonical, (200, 200))
    output_dir = tmp_path / "Maximum_Quality_Final"
    output_dir.mkdir()
    output = output_dir / "2-scene_enhanced.tif"
    output.write_bytes(b"alternate result")
    result = _run_cli(tmp_path, "--cleanup")

    assert result.returncode == 0, result.stderr
    assert not output.exists()
    assert (output_dir / "_archived_duplicates" / output.name).read_bytes() == b"alternate result"
