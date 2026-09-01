"""Contracts for deprecated inert Lux output flags (#2067)."""

from __future__ import annotations

import warnings
from dataclasses import asdict, fields
from pathlib import Path
from typing import Literal, get_type_hints

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
    "docs/validation/APEX_REAL_CANONICAL_EVIDENCE_RUNBOOK.md",
    "config/materials_v3_production.yaml",
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


@pytest.mark.parametrize(
    ("kwargs", "expected_depth", "expected_warning_count"),
    [
        ({}, 8, 0),
        ({"output_bit_depth": 8}, 8, 0),
        ({"output_bit_depth": 16}, 16, 0),
        ({"emit_master16": True}, 16, 1),
        ({"emit_upscaled16": True}, 16, 1),
        ({"emit_master16": True, "emit_upscaled16": True}, 16, 1),
        ({"emit_master16": False}, 8, 1),
        ({"output_bit_depth": 16, "emit_master16": True}, 16, 1),
        ({"output_bit_depth": 16, "emit_master16": True, "emit_upscaled16": True}, 16, 1),
    ],
)
def test_output_bit_depth_compatibility_matrix(kwargs, expected_depth, expected_warning_count) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = EnhanceConfig(**kwargs)

    bit_depth_warnings = [item for item in caught if isinstance(item.message, DeprecatedOutputFlagWarning)]
    assert config.output_bit_depth == expected_depth
    assert len(bit_depth_warnings) == expected_warning_count
    assert len(deprecated_output_flag_notices(config)) == expected_warning_count


def test_output_bit_depth_has_exact_public_type_and_default() -> None:
    assert get_type_hints(EnhanceConfig)["output_bit_depth"] == Literal[8, 16]
    output_field = next(field for field in fields(EnhanceConfig) if field.name == "output_bit_depth")
    assert output_field.default == 8
    assert EnhanceConfig().output_bit_depth == 8


@pytest.mark.parametrize("invalid", [None, True, False, 8.0, 8.9, "8", "16"])
def test_output_bit_depth_rejects_values_outside_the_typed_contract(invalid) -> None:
    with pytest.raises(ValueError, match="output_bit_depth must be 8 or 16"):
        EnhanceConfig(output_bit_depth=invalid)


@pytest.mark.parametrize(
    "aliases",
    [
        {"emit_master16": True},
        {"emit_upscaled16": True},
        {"emit_master16": True, "emit_upscaled16": True},
    ],
)
def test_explicit_8_bit_conflicts_with_truthy_legacy_aliases(aliases) -> None:
    with pytest.raises(ValueError, match="output_bit_depth=8 conflicts"):
        EnhanceConfig(output_bit_depth=8, **aliases)


def test_legacy_fingerprint_payload_reads_as_canonical_without_rewriting_aliases() -> None:
    from transformation_portal.lux_depth_v3.manifest import ConfigFingerprint

    fields = {
        "model_variant": "model",
        "depth_quantization": "u16",
        "depth_device": "cpu",
    }
    legacy = ConfigFingerprint(**fields, emit_master16=True, emit_upscaled16=False)
    canonical = ConfigFingerprint(**fields, output_bit_depth=16)

    assert legacy.output_bit_depth == 16
    assert legacy.to_sha256() == canonical.to_sha256()
    assert "emit_master16" not in asdict(legacy)
    assert "emit_upscaled16" not in asdict(legacy)


@pytest.mark.parametrize("invalid", [True, 8.0, 8.9, "8"])
def test_config_fingerprint_rejects_non_integer_output_depth(invalid) -> None:
    from transformation_portal.lux_depth_v3.manifest import ConfigFingerprint

    with pytest.raises(ValueError, match="output_bit_depth must be 8 or 16"):
        ConfigFingerprint(
            model_variant="model",
            depth_quantization="u16",
            depth_device="cpu",
            output_bit_depth=invalid,
        )


def test_maintained_surfaces_do_not_promise_fictional_16_bit_deliverables() -> None:
    forbidden = (
        "output/master16/",
        "output/upscaled16/",
        "Master16, Upscaled16",
        "_materials_v3_master16.",
        "_v2_master16.",
        '--emit-master16 "on"',
        '--emit-upscaled16 "on"',
        "--emit-master16 on",
        "--emit-upscaled16 on",
    )
    for relative_path in MAINTAINED_OPERATOR_SURFACES:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for stale_claim in forbidden:
            assert stale_claim not in text, f"{relative_path} still contains {stale_claim!r}"

    production_config = (REPO_ROOT / "config/materials_v3_production.yaml").read_text(encoding="utf-8")
    assert "emit_master16:" not in production_config
    assert "emit_upscaled16:" not in production_config
