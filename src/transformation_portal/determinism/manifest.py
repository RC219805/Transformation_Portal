from __future__ import annotations

import json
from typing import Any, Dict, Optional


def build_artifact_manifest(
    *,
    artifact_id: str,
    tensor_role: str,
    tensor_hash: str,
    raw_hash: str,
    fingerprint_hash: str,
    fpstate_enforced: bool,
    fpstate_backend: str,
    subnormals_preserved: bool,
    fpstate_note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Minimal deterministic artifact manifest.

    Rules:
      - No timestamps
      - No environment / host identifiers
      - Stable key ordering (serialize with stable_manifest_json)
      - Only content-derived + deterministic probe/enforcement outcomes
    """
    m: Dict[str, Any] = {
        "schema_version": 2,
        "artifact_id": artifact_id,
        "tensor_role": tensor_role,
        "tensor_hash": tensor_hash,
        "raw_input_hash": raw_hash,
        "fingerprint_hash": fingerprint_hash,
        "fpstate": {
            "enforced": bool(fpstate_enforced),
            "backend": str(fpstate_backend),
            "subnormals_preserved": bool(subnormals_preserved),
        },
    }
    if fpstate_note:
        # keep deterministic + short; no host data
        m["fpstate"]["note"] = str(fpstate_note)
    return m


def stable_manifest_json(manifest: Dict[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
