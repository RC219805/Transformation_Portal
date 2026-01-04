"""Batch statistics utilities.

Small, dependency-light helpers used by both the orchestrator and CLI.
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_batch_runtime_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute batch runtime stats from per-image result dicts.

    Expected result fields (best-effort):
      - status: "ok" | "error" | "skipped" | ...
      - runtime_s: float seconds (optional)

    Returns:
      Dict with:
        - total_runtime_s
        - avg_runtime_s (avg over ok results)
        - images_per_hour (throughput over ok results)
    """
    ok = sum(1 for r in results if r.get("status") == "ok")
    total_runtime_s = sum(float(r.get("runtime_s", 0.0) or 0.0) for r in results)
    avg_runtime_s = (total_runtime_s / ok) if ok else 0.0
    images_per_hour = ((ok / total_runtime_s) * 3600.0) if total_runtime_s > 0 else 0.0

    return {
        "total_runtime_s": total_runtime_s,
        "avg_runtime_s": avg_runtime_s,
        "images_per_hour": images_per_hour,
    }
