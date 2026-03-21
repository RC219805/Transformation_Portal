"""Tests for LLaVA prompt specifications."""

from __future__ import annotations

import pytest

from transformation_portal.evals.vision_language.llava_prompts import (
    LlavaPromptSpec,
    build_architectural_quality_prompt,
    build_depth_quality_prompt,
    build_segmentation_quality_prompt,
)

pytestmark = pytest.mark.unit


class TestLlavaPromptSpec:
    """Tests for LlavaPromptSpec dataclass."""

    def test_basic_construction(self) -> None:
        """LlavaPromptSpec should store name, system_text, and user_text."""
        spec = LlavaPromptSpec(
            name="test_prompt",
            system_text="You are an evaluator.",
            user_text="Evaluate this image.",
        )
        assert spec.name == "test_prompt"
        assert spec.system_text == "You are an evaluator."
        assert spec.user_text == "Evaluate this image."

    def test_is_frozen(self) -> None:
        """LlavaPromptSpec should be frozen (immutable)."""
        spec = LlavaPromptSpec(
            name="test",
            system_text="sys",
            user_text="user",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.name = "other"  # type: ignore


class TestSegmentationQualityPrompt:
    """Tests for build_segmentation_quality_prompt."""

    def test_basic_prompt(self) -> None:
        """Should build valid segmentation quality prompt."""
        prompt = build_segmentation_quality_prompt()
        assert prompt.name == "segmentation_mask_quality"
        assert "segmentation" in prompt.user_text.lower()
        assert "JSON" in prompt.system_text

    def test_with_context(self) -> None:
        """Should include context in prompt when provided."""
        context = {"source": "test_image.png", "stage": "reconstruction"}
        prompt = build_segmentation_quality_prompt(context=context)
        assert "Additional context" in prompt.user_text
        assert "test_image.png" in prompt.user_text


class TestArchitecturalQualityPrompt:
    """Tests for build_architectural_quality_prompt."""

    def test_basic_prompt(self) -> None:
        """Should build valid architectural quality prompt."""
        prompt = build_architectural_quality_prompt()
        assert prompt.name == "architectural_quality"
        assert "architectural" in prompt.user_text.lower()
        assert "perspective" in prompt.user_text.lower()

    def test_with_context(self) -> None:
        """Should include context in prompt when provided."""
        context = {"room_type": "living_room"}
        prompt = build_architectural_quality_prompt(context=context)
        assert "Additional context" in prompt.user_text


class TestDepthQualityPrompt:
    """Tests for build_depth_quality_prompt."""

    def test_basic_prompt(self) -> None:
        """Should build valid depth quality prompt."""
        prompt = build_depth_quality_prompt()
        assert prompt.name == "depth_map_quality"
        assert "depth" in prompt.user_text.lower()
        assert "bleeding" in prompt.user_text.lower()

    def test_with_context(self) -> None:
        """Should include context in prompt when provided."""
        context = {"backend": "depth_anything_v2"}
        prompt = build_depth_quality_prompt(context=context)
        assert "Additional context" in prompt.user_text
