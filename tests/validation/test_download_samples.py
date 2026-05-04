"""Unit tests for the synthetic-fixture path in scripts/download_samples.py."""

# pylint: disable=redefined-outer-name  # pytest fixtures

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

pytestmark = [
    pytest.mark.unit,
]


def _load_download_samples_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "download_samples.py"
    spec = importlib.util.spec_from_file_location("download_samples_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def download_samples() -> ModuleType:
    return _load_download_samples_module()


def test_rgb_gradient_dimensions_and_mode(download_samples: ModuleType, tmp_path: Path) -> None:
    output_path = tmp_path / "rgb.jpg"
    status = download_samples._generate_synthetic_image("rgb_gradient", output_path, width=100, height=100, seed=0)
    assert status == "generated"
    with Image.open(output_path) as img:
        assert img.size == (100, 100)
        assert img.mode == "RGB"


def test_depth_gradient_dimensions_and_mode(download_samples: ModuleType, tmp_path: Path) -> None:
    output_path = tmp_path / "depth.jpg"
    status = download_samples._generate_synthetic_image("depth_gradient", output_path, width=256, height=256, seed=1)
    assert status == "generated"
    with Image.open(output_path) as img:
        assert img.size == (256, 256)
        assert img.mode == "L"


def test_synthetic_output_is_deterministic(download_samples: ModuleType, tmp_path: Path) -> None:
    """Same kwargs must produce byte-identical fixtures across runs."""
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    assert download_samples._generate_synthetic_image("rgb_gradient", a, width=100, height=100, seed=0) == "generated"
    assert download_samples._generate_synthetic_image("rgb_gradient", b, width=100, height=100, seed=0) == "generated"
    assert a.read_bytes() == b.read_bytes()


def test_idempotent_when_existing_bytes_match(download_samples: ModuleType, tmp_path: Path) -> None:
    """Re-running with an up-to-date fixture must report 'up_to_date' and leave bytes/mtime untouched."""
    output_path = tmp_path / "rgb.jpg"
    assert (
        download_samples._generate_synthetic_image("rgb_gradient", output_path, width=100, height=100, seed=0) == "generated"
    )
    original_bytes = output_path.read_bytes()
    original_mtime_ns = output_path.stat().st_mtime_ns

    status = download_samples._generate_synthetic_image("rgb_gradient", output_path, width=100, height=100, seed=0)
    assert status == "up_to_date"
    assert output_path.read_bytes() == original_bytes
    assert output_path.stat().st_mtime_ns == original_mtime_ns


def test_stale_placeholder_fixture_is_migrated(download_samples: ModuleType, tmp_path: Path) -> None:
    """Pre-existing fixture content from older script versions must be replaced."""
    output_path = tmp_path / "rgb.jpg"
    # Simulate a stale placeholder fixture left behind by the old via.placeholder.com path.
    output_path.write_bytes(b"stale-placeholder-content")

    status = download_samples._generate_synthetic_image("rgb_gradient", output_path, width=100, height=100, seed=0)
    assert status == "generated"
    assert output_path.read_bytes() != b"stale-placeholder-content"
    with Image.open(output_path) as img:
        assert img.size == (100, 100)
        assert img.mode == "RGB"


def test_unwritable_directory_returns_failed_without_partial_output(download_samples: ModuleType, tmp_path: Path) -> None:
    """An unwritable target must yield status='failed' and leave no .tmp residue."""
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"i am a file, not a directory")
    # Path traversal through a regular file -> mkdir/write raises OSError.
    target = blocker / "nested" / "rgb.jpg"

    status = download_samples._generate_synthetic_image("rgb_gradient", target, width=8, height=8, seed=0)
    assert status == "failed"
    # No partial fixture or .tmp file should be left in the parent.
    leftover = list(tmp_path.glob("**/*.tmp"))
    assert not leftover, f"Unexpected .tmp residue: {leftover}"


def test_unknown_synthetic_kind_returns_failed(download_samples: ModuleType, tmp_path: Path) -> None:
    output_path = tmp_path / "x.jpg"
    status = download_samples._generate_synthetic_image("not_a_real_kind", output_path)
    assert status == "failed"
    assert not output_path.exists()
