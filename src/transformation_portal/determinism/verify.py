from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .policy import Adr030PolicyV1


@dataclass(frozen=True)
class VerificationResult:
    status: str
    max_abs_diff: float
    mae: float
    rmse: float
    pixel_parity_pass: bool
    global_drift_pass: bool

    def to_dict(self, *, policy: Adr030PolicyV1, baseline_artifact: str, candidate_artifact: str) -> Dict[str, Any]:
        return {
            "status": self.status,
            "baseline_artifact": baseline_artifact,
            "candidate_artifact": candidate_artifact,
            "verification_policy_version": policy.verification_policy_version,
            "policy_source": policy.policy_source,
            "gates": {
                "pixel_parity": {
                    "pass": self.pixel_parity_pass,
                    "max_abs_diff": self.max_abs_diff,
                    "bound": policy.max_abs_diff_bound,
                    "multiplier": policy.pixel_parity_multiplier,
                    "epsilon": policy.float32_eps,
                },
                "global_drift": {
                    "pass": self.global_drift_pass,
                    "mae": self.mae,
                    "rmse": self.rmse,
                    "mae_threshold": policy.mae_threshold,
                    "rmse_threshold": policy.rmse_threshold,
                },
            },
        }


def _assert_finite(arr: np.ndarray, name: str) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")


def compute_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")
    if reference.dtype != np.float32 or candidate.dtype != np.float32:
        raise ValueError("reference and candidate must be float32")
    _assert_finite(reference, "reference")
    _assert_finite(candidate, "candidate")

    ref64 = reference.astype(np.float64, copy=False).ravel(order="C")
    cand64 = candidate.astype(np.float64, copy=False).ravel(order="C")

    diff = ref64 - cand64
    abs_diff = np.abs(diff)

    max_abs_diff = float(np.max(abs_diff))
    mae = float(np.sum(abs_diff, dtype=np.float64) / abs_diff.size)
    rmse = float(np.sqrt(np.sum(diff * diff, dtype=np.float64) / diff.size))
    return max_abs_diff, mae, rmse


def verify_against_policy(reference: np.ndarray, candidate: np.ndarray, policy: Adr030PolicyV1) -> VerificationResult:
    max_abs_diff, mae, rmse = compute_metrics(reference, candidate)

    pixel_parity_pass = max_abs_diff <= policy.max_abs_diff_bound
    global_drift_pass = (mae < policy.mae_threshold) and (rmse < policy.rmse_threshold)

    status = "pass" if (pixel_parity_pass and global_drift_pass) else "fail"
    return VerificationResult(
        status=status,
        max_abs_diff=max_abs_diff,
        mae=mae,
        rmse=rmse,
        pixel_parity_pass=pixel_parity_pass,
        global_drift_pass=global_drift_pass,
    )
