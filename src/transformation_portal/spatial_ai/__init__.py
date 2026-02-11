"""Spatial AI Foundation — Linear Light Preservation & Geometric AI.

WARNING: This module is for training/research pipelines ONLY.
DO NOT use for rendering (outputs linear light, not display-ready).

For rendering, use: lux_depth_v3.raw_loader

See ADR-023 (Spatial AI Ingest Isolation Boundary) for architectural rationale.
"""

from __future__ import annotations

__all__ = ["ingest"]
