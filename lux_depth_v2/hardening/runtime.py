from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return "unknown"


def get_git_commit() -> str:
    # Prefer CI-provided env var; fallback to git if available.
    for k in ("GITHUB_SHA", "GIT_COMMIT", "CI_COMMIT_SHA"):
        v = os.getenv(k)
        if v:
            return v[:12]
    return _run(["git", "rev-parse", "HEAD"])[:12]


def get_runtime_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": get_git_commit(),
        "env": {
            "user": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
            "ci": bool(os.getenv("CI")),
        },
    }

    try:
        import torch  # type: ignore

        info["torch"] = {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
    except Exception:
        info["torch"] = {"version": "not_installed"}

    return info
