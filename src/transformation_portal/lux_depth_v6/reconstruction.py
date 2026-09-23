"""Versioned depth finishing reconstructed from independently verified V5 inputs.

This module consumes original photographic pixels and preserved native evidence,
never a previously finished image. Nonopaque input abstains because the upstream
V5 model proxy has no alpha-aware compositing contract.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from transformation_portal.core.depth_evidence import DepthEvidence
from transformation_portal.core.image_artifact import ImageMaster, ImageProxy, artifact_content_hash
from transformation_portal.lux_depth_v5.photography import AlignedDepth, align_depth, enhance_master_v5

from .depth_maps import DepthMapRecipe, reconstruct_depth


def _response_setting(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1 or not np.isfinite(value):
        raise ValueError("Upstream strength and clarity must be finite numbers in [0,1]")
    return float(value)


def reconstruct_depth_baseline(
    original: ImageMaster,
    evidence: DepthEvidence,
    proxy: ImageProxy,
    configuration: Mapping[str, Any],
    recipe: DepthMapRecipe,
) -> tuple[ImageMaster, dict[str, Any], AlignedDepth, dict[str, Any]]:
    """Use one frozen alignment for both photographic response and scalar maps."""
    if not isinstance(configuration, Mapping) or not isinstance(configuration.get("depth"), Mapping):
        raise ValueError("Reconstruction requires the verified upstream depth configuration")
    depth = configuration["depth"]
    if depth.get("refinement") not in ("bilinear", "guided_bilinear") or depth.get("precision") != evidence.precision:
        raise ValueError("Upstream refinement or precision does not authorize this depth evidence")
    strength = _response_setting(configuration.get("strength"))
    clarity = _response_setting(configuration.get("clarity"))
    aligned, depth_receipt = reconstruct_depth(original, evidence, proxy, recipe)
    baseline, response = original, None
    if not depth_receipt["alpha_abstention"]:
        baseline, response = enhance_master_v5(original, aligned, strength=strength, clarity=clarity)
    receipt = {
        **depth_receipt,
        "schema": "tp.lux.depth_reconstruction.v2",
        "recipe": "shared_depth_map_reconstruction_v1",
        "upstream_refinement": depth["refinement"],
        "refinement": recipe.refinement,
        "strength": strength,
        "clarity": clarity,
        "depth_response": response,
        "baseline_content_sha256": baseline.content_hash(),
    }
    return baseline, receipt, aligned, depth_receipt


def reconstruct_baseline(
    original: ImageMaster, evidence: DepthEvidence, proxy: ImageProxy, configuration: Mapping[str, Any]
) -> tuple[ImageMaster, dict[str, Any]]:
    """Retain bounded upstream edit settings while requiring persistent surfaces.

    The caller must independently verify the V5 execution evidence before use.
    The input bindings and configuration are checked again here, including when
    alpha requires abstention. This proves recipe execution, not scene accuracy.
    """
    if not isinstance(original, ImageMaster) or not isinstance(evidence, DepthEvidence) or not isinstance(proxy, ImageProxy):
        raise ValueError("Reconstruction requires typed original, native depth, and proxy evidence")
    if not isinstance(configuration, Mapping) or not isinstance(configuration.get("depth"), Mapping):
        raise ValueError("Reconstruction requires the verified upstream depth configuration")
    depth = configuration["depth"]
    upstream_refinement = depth.get("refinement")
    if upstream_refinement not in ("bilinear", "guided_bilinear") or depth.get("precision") != evidence.precision:
        raise ValueError("Upstream refinement or precision does not authorize this depth evidence")
    strength = _response_setting(configuration.get("strength"))
    clarity = _response_setting(configuration.get("clarity"))
    original_hash = original.content_hash()
    proxy_hash = artifact_content_hash(
        {"transform": proxy.transform.to_payload(), "master_content_hash": proxy.master_content_hash},
        {"pixels": proxy.pixels},
    )
    if (
        evidence.source_sha256 != original.source_sha256
        or evidence.transform != proxy.transform
        or proxy.transform.original_shape != original.shape
        or proxy.master_content_hash != original_hash
        or evidence.metadata.get("proxy_master_content_hash") != original_hash
        or evidence.metadata.get("proxy_content_hash") != proxy_hash
    ):
        raise ValueError("Reconstruction inputs must bind the same source, master, and proxy geometry")
    refinement = "guided_bilinear_v3" if upstream_refinement == "guided_bilinear" else "bilinear"
    alpha_abstention = original.alpha is not None and bool(np.any(original.alpha != 1))
    receipt: dict[str, Any] = {
        "schema": "tp.lux.depth_reconstruction.v1",
        "recipe": "persistent_native_reconstruction_v1",
        "source_sha256": original.source_sha256,
        "original_content_sha256": original_hash,
        "depth_evidence_content_sha256": evidence.content_hash(),
        "proxy_content_sha256": proxy_hash,
        "upstream_refinement": upstream_refinement,
        "refinement": refinement,
        "strength": float(strength),
        "clarity": float(clarity),
        "alpha_abstention": alpha_abstention,
        "reason": "alpha_unaware_upstream_proxy" if alpha_abstention else None,
        "aligned_content_sha256": None,
        "depth_response": None,
        "quality_acceptance": "unestablished",
    }
    baseline = original
    if not alpha_abstention:
        aligned = align_depth(evidence, original, proxy, refinement=refinement)
        baseline, response = enhance_master_v5(original, aligned, strength=float(strength), clarity=float(clarity))
        receipt["aligned_content_sha256"] = aligned.content_hash()
        receipt["depth_response"] = response
    receipt["baseline_content_sha256"] = baseline.content_hash()
    return baseline, receipt
