"""Tests for CLI model variant string parsing."""

import pytest
from lux_depth_v3.cli import parse_model_variant, MODEL_VARIANT_MAP
from lux_depth_v3.config import ModelVariant


class TestParseModelVariant:
    """Test suite for parse_model_variant function."""

    def test_parse_metric_large(self):
        """Test parsing of metric-large model name."""
        result = parse_model_variant("metric-large")
        assert result == ModelVariant.METRIC_LARGE

    def test_parse_nested_giant_large(self):
        """Test parsing of nested-giant-large-v1.1 model name."""
        result = parse_model_variant("nested-giant-large-v1.1")
        assert result == ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1

    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        result = parse_model_variant("METRIC-LARGE")
        assert result == ModelVariant.METRIC_LARGE

    def test_parse_legacy_uppercase(self):
        """Test parsing of legacy uppercase variants."""
        result = parse_model_variant("METRIC_LARGE")
        assert result == ModelVariant.METRIC_LARGE

    def test_parse_base_model(self):
        """Test parsing of base model (Apache 2.0)."""
        result = parse_model_variant("base")
        assert result == ModelVariant.DA3_BASE

    def test_parse_invalid_raises_exit(self):
        """Test that invalid model name raises typer.Exit."""
        import typer
        with pytest.raises(typer.Exit):
            parse_model_variant("invalid-model-name")

    def test_all_variants_in_map(self):
        """Ensure all common model variants are in the mapping."""
        required_names = [
            "metric-large",
            "nested-giant-large-v1.1",
            "giant-v1.1",
            "large-v1.1",
            "base",
            "small",
        ]
        for name in required_names:
            assert name in MODEL_VARIANT_MAP, f"{name} should be in MODEL_VARIANT_MAP"

    def test_model_variant_map_values_are_enums(self):
        """Ensure all values in MODEL_VARIANT_MAP are ModelVariant enums."""
        for key, value in MODEL_VARIANT_MAP.items():
            assert isinstance(value, ModelVariant), f"{key} maps to non-ModelVariant: {value}"
