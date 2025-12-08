from pathlib import Path

import pytest

from lux_depth_v2.hardening.policy import HardeningPolicy
from lux_depth_v2.hardening.safe_io import sniff_image_type, validate_image_file


def _write(path: Path, b: bytes) -> None:
    path.write_bytes(b)


def test_sniff_tiff(tmp_path: Path):
    f = tmp_path / "x.tif"
    _write(f, b"II*\x00" + b"\x00" * 32)
    assert sniff_image_type(f) == "tiff"


def test_sniff_png(tmp_path: Path):
    f = tmp_path / "x.png"
    _write(f, b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    assert sniff_image_type(f) == "png"


def test_validate_rejects_missing(tmp_path: Path):
    policy = HardeningPolicy()
    with pytest.raises(Exception):
        validate_image_file(tmp_path / "missing.tif", policy)


def test_validate_rejects_extension(tmp_path: Path):
    policy = HardeningPolicy()
    f = tmp_path / "x.exe"
    _write(f, b"MZ" + b"\x00" * 32)
    with pytest.raises(Exception):
        validate_image_file(f, policy)


def test_validate_rejects_oversize(tmp_path: Path):
    policy = HardeningPolicy(max_input_bytes=10)
    f = tmp_path / "x.tif"
    _write(f, b"II*\x00" + b"\x00" * 64)
    with pytest.raises(Exception):
        validate_image_file(f, policy)


def test_validate_accepts_small_tiff(tmp_path: Path):
    policy = HardeningPolicy(max_input_bytes=10_000)
    f = tmp_path / "x.tif"
    _write(f, b"II*\x00" + b"\x00" * 64)
    validate_image_file(f, policy)  # no exception
