"""Contract tests for repository Git attributes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GITATTRIBUTES_PATH = PROJECT_ROOT / ".gitattributes"


TEXT_FIXTURES = [
    ".gitattributes",
    ".gitignore",
    ".github/CODEOWNERS",
    ".env.example",
    "Dockerfile",
    "Makefile",
    "README.md",
    "wrangler.jsonc",
    "migrations/script.py.mako",
    "docs/contracts/SCHEMA_LOCKS.sha256",
    "docs/contracts/presence/PresenceCompiler.sol",
    "policy/dataset_signing_public.asc",
    "src/transformation_portal/determinism/_fpstate.c",
    "tools/verify_merkle_signature.py",
]

BINARY_FIXTURES = [
    "public/portal-assets/fonts/portal-sans.woff2",
    "src/transformation_portal/lux_depth_v3_precision_arch_fixes_plus_upgrades_files.zip",
    "tests/fixtures/archive_small/archive_index_normalized.csv.gz",
    "tests/fixtures/archive_small/archive_root/DriveA/Part1/bravo.bin",
    "tests/fixtures/ingest/batch_inputs/raw/DJI_0001.DNG",
    "tests/fixtures/ingest/batch_inputs/stills/INT_1001.JPG",
    "tests/fixtures/ingest/batch_inputs/video/clip_0001.MOV",
    "tests/fixtures/pipelines/750_picacho_lane/input/750Picacho_Pool_UltraQuality.tif",
]


def _git_attrs(path: str) -> dict[str, str]:
    result = subprocess.run(
        ["git", "check-attr", "--all", "--", path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    attrs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        _path, attr, value = line.split(": ", 2)
        attrs[attr] = value
    return attrs


def test_gitattributes_has_no_duplicate_exact_patterns() -> None:
    seen: dict[str, int] = {}

    for line_number, line in enumerate(GITATTRIBUTES_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pattern = stripped.split()[0]
        assert pattern not in seen, f"duplicate pattern {pattern!r} at lines {seen[pattern]} and {line_number}"
        seen[pattern] = line_number


@pytest.mark.parametrize("path", TEXT_FIXTURES)
def test_representative_text_files_are_lf_normalized(path: str) -> None:
    assert (PROJECT_ROOT / path).exists(), path
    attrs = _git_attrs(path)

    assert attrs["text"] == "set"
    assert attrs["eol"] == "lf"


@pytest.mark.parametrize("path", BINARY_FIXTURES)
def test_representative_binary_files_are_not_text_normalized(path: str) -> None:
    assert (PROJECT_ROOT / path).exists(), path
    attrs = _git_attrs(path)

    assert attrs["binary"] == "set"
    assert attrs["text"] == "unset"
    assert attrs["diff"] == "unset"
    assert attrs["merge"] == "unset"
