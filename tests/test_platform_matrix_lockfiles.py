from pathlib import Path

import pytest

from transformation_portal.core.execution_identity import resolve_platform_lockfile
from transformation_portal.core.platform_matrix import (
    PlatformAccel,
    PlatformISA,
    PlatformMatrix,
    PlatformOS,
    determine_ml_core_lockfile_name,
)

pytestmark = pytest.mark.unit


def test_determine_ml_core_lockfile_name_for_darwin_intel() -> None:
    matrix = PlatformMatrix(PlatformOS.DARWIN, PlatformISA.X86_64, PlatformAccel.CPU)
    assert determine_ml_core_lockfile_name(matrix) == "ml-core-darwin-x86_64.txt"


def test_determine_ml_core_lockfile_name_for_darwin_arm64() -> None:
    matrix = PlatformMatrix(PlatformOS.DARWIN, PlatformISA.ARM64, PlatformAccel.MPS)
    assert determine_ml_core_lockfile_name(matrix) == "ml-core-darwin-arm64.txt"


def test_determine_ml_core_lockfile_name_for_linux_arm64() -> None:
    matrix = PlatformMatrix(PlatformOS.LINUX, PlatformISA.ARM64, PlatformAccel.CPU)
    assert determine_ml_core_lockfile_name(matrix) == "ml-core-linux.txt"


def test_resolve_platform_lockfile_returns_split_darwin_name() -> None:
    result = resolve_platform_lockfile()
    assert result is None or isinstance(result, Path)
    if result is not None and "darwin" in result.name:
        assert result.name in {"ml-core-darwin-x86_64.txt", "ml-core-darwin-arm64.txt"}


def test_linux_ml_security_posture_reports_frozen_historical_lane() -> None:
    matrix = PlatformMatrix(PlatformOS.LINUX, PlatformISA.X86_64, PlatformAccel.CPU)

    status = matrix.check_ml_security_posture()

    assert status["ml_supported"] is False
    assert status["secure"] is False
    assert "frozen unsupported historical lane" in status["cve_2025_32434_note"]
