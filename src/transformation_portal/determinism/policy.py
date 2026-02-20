from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .jcs import sha256_hex_of_canonical_json


@dataclass(frozen=True)
class Adr030PolicyV1:
    verification_policy_version: str
    float_model: str
    float32_eps_exp: int
    pixel_parity_multiplier: int
    mae_threshold: float
    rmse_threshold: float
    nan_policy: str
    inf_policy: str
    subnormal_policy: str
    ftz_daz_policy: str
    reduction_mode: str
    matrix_backend: str
    certified_tensor_role: str
    policy_source: str

    @property
    def float32_eps(self) -> float:
        # Exact for integer exponent.
        return 2.0 ** float(self.float32_eps_exp)

    @property
    def max_abs_diff_bound(self) -> float:
        return float(self.pixel_parity_multiplier) * self.float32_eps


def _require(obj: Dict[str, Any], key: str) -> Any:
    if key not in obj:
        raise KeyError(f"Policy missing required key: {key}")
    return obj[key]


def load_policy(path: Path) -> tuple[Adr030PolicyV1, str]:
    """Load ADR-030 policy from JSON.

    Returns (policy, policy_hash_hex) where policy_hash_hex is SHA-256 of the
    RFC 8785 canonical JSON bytes (JCS).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    policy_hash = sha256_hex_of_canonical_json(raw)

    policy = Adr030PolicyV1(
        verification_policy_version=str(_require(raw, "verification_policy_version")),
        float_model=str(_require(raw, "float_model")),
        float32_eps_exp=int(_require(raw, "float32_eps_exp")),
        pixel_parity_multiplier=int(_require(raw, "pixel_parity_multiplier")),
        mae_threshold=float(_require(raw, "mae_threshold")),
        rmse_threshold=float(_require(raw, "rmse_threshold")),
        nan_policy=str(_require(raw, "nan_policy")),
        inf_policy=str(_require(raw, "inf_policy")),
        subnormal_policy=str(_require(raw, "subnormal_policy")),
        ftz_daz_policy=str(_require(raw, "ftz_daz_policy")),
        reduction_mode=str(_require(raw, "reduction_mode")),
        matrix_backend=str(_require(raw, "matrix_backend")),
        certified_tensor_role=str(_require(raw, "certified_tensor_role")),
        policy_source=str(_require(raw, "policy_source")),
    )
    return policy, policy_hash
