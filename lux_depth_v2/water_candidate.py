"""
Water Candidate Detection Module - STUB IMPLEMENTATION

⚠️ WARNING: This is a minimal stub implementation for testing.
Status: PR-W1 pending full multi-cue heuristic detector.

This stub provides a simple blue-threshold detector to enable PR-W4 validation
harness testing. For production use, this should be replaced with the full
implementation described in docs/PR_WATER_MASK_STRUCTURE.md PR-W1 section.

The full PR-W1 implementation should include:
- Multi-cue heuristics (chromaticity, specular, texture, planarity)
- Scene-aware tuning (pool vs ocean)
- Post-processing pipeline
- Component filtering

Location: lux_depth_v2/water_candidate.py (stub)
DO NOT rely on this for production water detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class SceneContext(str, Enum):
    """Scene context for water detection heuristics."""
    POOL = "pool"
    OCEAN = "ocean"
    UNKNOWN = "unknown"


@dataclass
class WaterHeuristicConfig:
    """Configuration for water heuristic detector."""
    blue_ratio_threshold: float = 0.4
    saturation_threshold: float = 0.2
    min_coverage_threshold: float = 0.05
    confidence_threshold: float = 0.5


class WaterCandidateDetector:
    """
    Heuristic-based water detector (stub for PR-W4 testing).

    This is a minimal stub implementation. The full detector
    should implement the heuristics described in PR-W1.
    """

    def __init__(self, config: Optional[WaterHeuristicConfig] = None):
        self.config = config or WaterHeuristicConfig()

    def detect(
        self,
        rgb01: np.ndarray,
        depth01: Optional[np.ndarray] = None,
        scene_context: SceneContext = SceneContext.UNKNOWN
    ) -> dict:
        """
        Detect water candidates in RGB image.

        Args:
            rgb01: RGB image (HxWx3 float32 in [0,1])
            depth01: Optional depth map (HxW float32 in [0,1]) - unused in stub
            scene_context: Scene context hint (unused in stub)

        Returns:
            dict with keys:
                - present: bool
                - coverage: float (0-1)
                - coverage_px: int
                - confidence: float (0-1)
                - mask: np.ndarray (H, W) binary mask
        """
        # Stub implementation: simple blue threshold (ignores depth and context)
        h, w = rgb01.shape[:2]

        # Detect blue-ish regions
        if rgb01.shape[2] == 3:
            blue = rgb01[:, :, 2]
            green = rgb01[:, :, 1]
            red = rgb01[:, :, 0]

            # Simple heuristic: blue channel dominant
            blue_dominant = (blue > red) & (blue > green * 0.8)
            blue_threshold = blue > 0.3

            mask = (blue_dominant & blue_threshold).astype(np.float32)
        else:
            mask = np.zeros((h, w), dtype=np.float32)

        # Calculate coverage
        coverage_px = int(np.sum(mask))
        total_px = h * w
        coverage = float(coverage_px / total_px) if total_px > 0 else 0.0

        # Calculate confidence (stub: just based on coverage)
        confidence = min(coverage * 2, 1.0) if coverage > 0 else 0.0

        # Determine if present
        present = coverage >= self.config.min_coverage_threshold and confidence >= self.config.confidence_threshold

        return {
            "present": present,
            "coverage": coverage,
            "coverage_px": coverage_px,
            "confidence": confidence,
            "mask": mask,
            "implementation": "stub_v0_blue_threshold"
        }
