"""Contracts for deprecated inert Lux output flags (#2067)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.config import (
    DeprecatedOutputFlagWarning,
    EnhanceConfig,
    deprecated_output_flag_notices,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
MAINTAINED_OPERATOR_SURFACES = (
    "src/transformation_portal/lux_depth_v3/__main__.py",
    "src/transformation_portal/lux_depth_v3/README.md",
    "docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md",
    "docs/cli/CLI_REFERENCE.md",
    "docs/guides/LUX_DEPTH_V3_TROUBLESHOOTING.md",
    "docs/guides/FILE_FORMAT_QUICK_REFERENCE.md",
    "docs/guides/IMAGE_PROCESSING_READINESS.md",
    "docs/guides/SUPPORTED_FILE_FORMATS.md",
    "docs/reference/QUICKSTART_CHEATSHEET.md",
    "scripts/pipelines/run_montecito_apex_full.sh",
    "scripts/pipelines/run_montecito_apex_lean.sh",
)


def test_omitted_legacy_flags_keep_historical_runtime_defaults_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = EnhanceConfig(model_key="da3-metric")

    assert config.emit_marketing is False
    assert config.emit_report is True
    assert not deprecated_output_flag_notices(config)
    assert not any(isinstance(item.message, DeprecatedOutputFlagWarning) for item in caught)


def test_maintained_surfaces_do_not_promise_or_recommend_inert_flags() -> None:
    forbidden = (
        "*_marketing.jpg",
        "marketing/",
        "Marketing Deliverables Only",
        "--emit-marketing on",
        '--emit-marketing "on"',
        "--emit-report on",
        '--emit-report "on"',
        "when --emit-report",
        "when `--emit-report",
    )

    for relative_path in MAINTAINED_OPERATOR_SURFACES:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for stale_claim in forbidden:
            assert stale_claim not in text, f"{relative_path} still contains {stale_claim!r}"


def test_deprecation_window_is_documented() -> None:
    guide = (REPO_ROOT / "docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md").read_text(encoding="utf-8")
    assert "--emit-marketing` and `--emit-report` are deprecated" in guide
    assert "next major release" in guide
    assert "combined processing\nmanifest is always emitted" in guide
