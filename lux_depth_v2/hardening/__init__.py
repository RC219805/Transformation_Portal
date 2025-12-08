"""
Lux Depth V2 - Architecture Hardening Layer (opt-in)

Goal: Add production-grade guardrails (security, reproducibility, observability)
WITHOUT changing existing pipeline behavior unless explicitly enabled.

Primary entrypoints:
- HardeningPolicy
- LuxPipelineV2Hardened (wrapper around LuxPipelineV2)
- create_hardened_app (FastAPI hardening wrapper factory)
"""

from .policy import HardeningPolicy
from .wrapper import LuxPipelineV2Hardened, HardenedRunResult
from .service_factory import create_hardened_app
from .safe_io import safe_resolve_under, redact_path

__all__ = [
    "HardeningPolicy",
    "LuxPipelineV2Hardened",
    "HardenedRunResult",
    "create_hardened_app",
    "safe_resolve_under",
    "redact_path",
]
