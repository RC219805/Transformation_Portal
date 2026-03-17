"""Prompt specifications for structured LLaVA visual quality assessment.

This module provides prompt templates for various quality assessment tasks
using LLaVA vision-language models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LlavaPromptSpec:
    """Specification for a LLaVA evaluation prompt.

    Attributes:
        name: Identifier for this prompt specification
        system_text: System message setting the evaluation context
        user_text: User message with the actual evaluation request
    """

    name: str
    system_text: str
    user_text: str


def build_segmentation_quality_prompt(
    context: Optional[dict[str, Any]] = None,
) -> LlavaPromptSpec:
    """Build a prompt for segmentation/reconstruction quality assessment.

    Args:
        context: Optional additional context to include in the prompt

    Returns:
        LlavaPromptSpec configured for segmentation quality evaluation
    """
    context_suffix = ""
    if context:
        context_suffix = f"\nAdditional context:\n{context}\n"

    return LlavaPromptSpec(
        name="segmentation_mask_quality",
        system_text=(
            "You are a strict computer vision quality assurance evaluator. "
            "You must return only valid JSON, with no surrounding markdown."
        ),
        user_text=(
            "Assess the provided image(s) for segmentation or reconstruction quality issues.\n"
            "Check specifically for:\n"
            "1. segmentation leakage into background or adjacent objects\n"
            "2. missing object regions or holes\n"
            "3. silhouette or geometry distortion\n"
            "4. obvious texture seams or discontinuities\n"
            "5. hallucinated details or implausible structures\n\n"
            "Return only valid JSON with this schema:\n"
            "{\n"
            '  "passes_basic_quality": boolean,\n'
            '  "summary_score": number,\n'
            '  "issues": [\n'
            "    {\n"
            '      "issue_type": string,\n'
            '      "severity": "low|medium|high",\n'
            '      "evidence": string\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"{context_suffix}"
        ),
    )


def build_architectural_quality_prompt(
    context: Optional[dict[str, Any]] = None,
) -> LlavaPromptSpec:
    """Build a prompt for architectural/real estate image quality assessment.

    Args:
        context: Optional additional context to include in the prompt

    Returns:
        LlavaPromptSpec configured for architectural quality evaluation
    """
    context_suffix = ""
    if context:
        context_suffix = f"\nAdditional context:\n{context}\n"

    return LlavaPromptSpec(
        name="architectural_quality",
        system_text=(
            "You are an expert architectural visualization quality assessor. "
            "You must return only valid JSON, with no surrounding markdown."
        ),
        user_text=(
            "Evaluate this architectural/real estate image for quality issues.\n"
            "Check specifically for:\n"
            "1. perspective distortion or incorrect vanishing points\n"
            "2. unnatural lighting or shadow inconsistencies\n"
            "3. material rendering artifacts (unrealistic reflections, textures)\n"
            "4. color banding or posterization\n"
            "5. visible compression artifacts or noise\n"
            "6. geometric errors (bent lines, warped surfaces)\n\n"
            "Return only valid JSON with this schema:\n"
            "{\n"
            '  "passes_basic_quality": boolean,\n'
            '  "summary_score": number,\n'
            '  "issues": [\n'
            "    {\n"
            '      "issue_type": string,\n'
            '      "severity": "low|medium|high",\n'
            '      "evidence": string\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"{context_suffix}"
        ),
    )


def build_depth_quality_prompt(
    context: Optional[dict[str, Any]] = None,
) -> LlavaPromptSpec:
    """Build a prompt for depth map quality assessment.

    Args:
        context: Optional additional context to include in the prompt

    Returns:
        LlavaPromptSpec configured for depth map quality evaluation
    """
    context_suffix = ""
    if context:
        context_suffix = f"\nAdditional context:\n{context}\n"

    return LlavaPromptSpec(
        name="depth_map_quality",
        system_text=(
            "You are an expert depth estimation quality assessor. "
            "You must return only valid JSON, with no surrounding markdown."
        ),
        user_text=(
            "Evaluate this depth map visualization for quality issues.\n"
            "Check specifically for:\n"
            "1. depth bleeding at object edges\n"
            "2. incorrect relative depth ordering\n"
            "3. missing depth detail in fine structures\n"
            "4. depth discontinuities in smooth surfaces\n"
            "5. artifacts in transparent or reflective regions\n\n"
            "Return only valid JSON with this schema:\n"
            "{\n"
            '  "passes_basic_quality": boolean,\n'
            '  "summary_score": number,\n'
            '  "issues": [\n'
            "    {\n"
            '      "issue_type": string,\n'
            '      "severity": "low|medium|high",\n'
            '      "evidence": string\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"{context_suffix}"
        ),
    )
