"""Presence Security v1.2 helpers."""

from transformation_portal.presence_security.countermeasures import (
    add_dither,
    randomized_blend_weights,
    randomized_eye_line,
    randomized_prompts,
)
from transformation_portal.presence_security.parameters import PresenceParameters
from transformation_portal.presence_security.watermarking import (
    embed_dct_luma,
    embed_lsb_rgb,
    extract_lsb_rgb,
    manifest_session_from_lsb,
    sha3_manifest_hex,
)

__all__ = [
    "PresenceParameters",
    "add_dither",
    "embed_dct_luma",
    "embed_lsb_rgb",
    "extract_lsb_rgb",
    "manifest_session_from_lsb",
    "randomized_blend_weights",
    "randomized_eye_line",
    "randomized_prompts",
    "sha3_manifest_hex",
]
