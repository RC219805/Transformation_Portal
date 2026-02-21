from __future__ import annotations

import re


def parse_artifact_id(artifact_id: str) -> str:
    """
    Validate and extract 64-char hex hash from artifact_id.

    Expected format:
        sha256:<64_hex>
    """
    if not artifact_id.startswith("sha256:"):
        raise ValueError("artifact_id must start with 'sha256:'")

    hex_part = artifact_id.split("sha256:", 1)[1].lower()

    if not re.fullmatch(r"[0-9a-f]{64}", hex_part):
        raise ValueError(f"artifact_id must contain 64 lowercase hex characters, got: {hex_part!r}")

    return hex_part
