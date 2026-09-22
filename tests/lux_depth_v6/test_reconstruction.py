"""Successor depth reconstruction and alpha abstention contracts."""

from dataclasses import replace

import numpy as np
import pytest

from transformation_portal.core.depth_evidence import build_depth_evidence
from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.lux_depth_v4.photography import create_proxy
from transformation_portal.lux_depth_v5.photography import align_depth, enhance_master_v5
from transformation_portal.lux_depth_v6.reconstruction import reconstruct_baseline

pytestmark = pytest.mark.unit


def fixture(alpha=None):
    pixels = np.full((56, 112, 3), 0.1, np.float32)
    pixels[:, 56:] = 0.8
    original = ImageMaster(pixels, "a" * 64, 16, alpha=alpha)
    proxy = create_proxy(original, 28)
    native = np.ones(proxy.transform.padded_shape, np.float32)
    native[:, 14:] = 2
    evidence = build_depth_evidence(native, np.zeros_like(native, bool), proxy, original.source_sha256)
    configuration = {"depth": {"precision": "fp32", "refinement": "guided_bilinear"}, "strength": 0.4, "clarity": 0.3}
    return original, evidence, proxy, configuration


@pytest.mark.parametrize("refinement", ["guided_bilinear", "bilinear"])
def test_baseline_reconstructs_requested_response_from_original_and_native(refinement):
    original, evidence, proxy, configuration = fixture()
    configuration["depth"]["refinement"] = refinement
    before = evidence.native_depth.tobytes()
    baseline, receipt = reconstruct_baseline(original, evidence, proxy, configuration)
    selected = "guided_bilinear_v3" if refinement == "guided_bilinear" else "bilinear"
    aligned = align_depth(evidence, original, proxy, refinement=selected)
    expected, response = enhance_master_v5(original, aligned, strength=0.4, clarity=0.3)
    assert baseline.content_hash() == expected.content_hash()
    assert evidence.native_depth.tobytes() == before
    assert receipt["depth_response"] == response
    assert receipt["refinement"] == selected
    assert receipt["upstream_refinement"] == refinement
    assert receipt["original_content_sha256"] == original.content_hash()
    assert receipt["depth_evidence_content_sha256"] == evidence.content_hash()
    assert receipt["baseline_content_sha256"] == baseline.content_hash()
    assert receipt["aligned_content_sha256"] == aligned.content_hash()
    assert receipt["quality_acceptance"] == "unestablished"
    assert not receipt["alpha_abstention"]


@pytest.mark.parametrize("alpha_value", [0.0, 0.5, np.nextafter(np.float32(1), np.float32(0))])
def test_any_nonopaque_pixel_abstains_from_all_depth_finishing(alpha_value):
    alpha = np.ones((56, 112), np.float32)
    alpha[0, 0] = alpha_value
    original, evidence, proxy, configuration = fixture(alpha)
    baseline, receipt = reconstruct_baseline(original, evidence, proxy, configuration)
    assert baseline is original
    assert baseline.pixels.tobytes() == original.pixels.tobytes()
    assert receipt["reason"] == "alpha_unaware_upstream_proxy"
    assert receipt["alpha_abstention"]
    assert receipt["depth_response"] is receipt["aligned_content_sha256"] is None


def test_fully_opaque_alpha_allows_depth_finishing():
    original, evidence, proxy, configuration = fixture(np.ones((56, 112), np.float32))
    baseline, receipt = reconstruct_baseline(original, evidence, proxy, configuration)
    assert not receipt["alpha_abstention"]
    response = receipt["depth_response"]
    assert isinstance(response, dict)
    assert dict(response)["changed_pixels"] > 0
    np.testing.assert_array_equal(baseline.alpha, original.alpha)


@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("changed", ["source", "pixels", "metadata", "proxy", "evidence"])
def test_bindings_are_checked_even_when_alpha_requires_abstention(alpha, changed):
    original, evidence, proxy, configuration = fixture(np.full((56, 112), 0.5, np.float32) if alpha else None)
    if changed == "source":
        original = replace(original, source_sha256="b" * 64)
    elif changed == "pixels":
        original = replace(original, pixels=original.pixels * 0.5)
    elif changed == "metadata":
        original = replace(original, metadata={"raw_decoder_recipe": "changed_after_verification"})
    elif changed == "proxy":
        proxy = replace(proxy, pixels=255 - proxy.pixels)
    else:
        evidence = replace(evidence, metadata={})
    with pytest.raises(ValueError, match="same source"):
        reconstruct_baseline(original, evidence, proxy, configuration)


@pytest.mark.parametrize("value", [None, True, -0.1, 1.1, float("inf"), float("nan"), "0.5"])
def test_invalid_upstream_response_configuration_fails_before_alpha_abstention(value):
    original, evidence, proxy, configuration = fixture(np.full((56, 112), 0.5, np.float32))
    configuration["strength"] = value
    with pytest.raises(ValueError, match="finite numbers"):
        reconstruct_baseline(original, evidence, proxy, configuration)


@pytest.mark.parametrize(
    "depth", [{}, {"precision": "fp16", "refinement": "bilinear"}, {"precision": "fp32", "refinement": "magic"}]
)
def test_unknown_upstream_depth_policy_fails_closed(depth):
    original, evidence, proxy, configuration = fixture()
    configuration["depth"] = depth
    with pytest.raises(ValueError, match="refinement or precision"):
        reconstruct_baseline(original, evidence, proxy, configuration)


def test_zero_response_retains_original_photographic_pixel_bits():
    original, evidence, proxy, configuration = fixture()
    configuration.update(strength=0, clarity=0)
    baseline, receipt = reconstruct_baseline(original, evidence, proxy, configuration)
    assert baseline.pixels.tobytes() == original.pixels.tobytes()
    response = receipt["depth_response"]
    assert isinstance(response, dict)
    assert dict(response)["changed_pixels"] == 0


def test_unknown_sky_evidence_remains_protected_in_successor():
    original, evidence, proxy, configuration = fixture()
    evidence = build_depth_evidence(evidence.native_depth, None, proxy, original.source_sha256)
    baseline, receipt = reconstruct_baseline(original, evidence, proxy, configuration)
    assert baseline.pixels.tobytes() == original.pixels.tobytes()
    response = receipt["depth_response"]
    assert isinstance(response, dict)
    assert dict(response)["authorized_pixels"] == 0
