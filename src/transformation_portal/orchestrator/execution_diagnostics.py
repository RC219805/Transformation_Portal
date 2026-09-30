"""Bounded, allowlisted recovery guidance for known photographic failures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from transformation_portal.orchestrator.job_execution import ExecutionJob

_PHOTOGRAPHY_PIPELINES = frozenset({"lux-depth", "lux-depth-v5", "lux-depth-v6"})
_ICC_GUIDANCE = (
    "unsupported_icc_profile",
    "Input uses an unsupported ICC profile. Convert the image to sRGB with a profile-aware editor and upload the converted file. "
    "If Auto rejects its sRGB profile, select sRGB only for that converted file, then create a new job. "
    "Do not relabel Adobe RGB as sRGB.",
)
_AMBIGUOUS_GUIDANCE = (
    "ambiguous_input_color",
    "Input color could not be determined. Select sRGB or linear sRGB only when it matches the source encoding, "
    "or convert the image to sRGB with an embedded profile, then create a new job.",
)
_FAILURE_GUIDANCE = {
    "Unsupported ICC profile; provide an explicit input_color correction": _ICC_GUIDANCE,
    (
        "Unsupported ICC profile. Convert the photographs to sRGB with color management and upload the converted "
        "copies. An input color override only reinterprets existing pixels; it does not convert them."
    ): _ICC_GUIDANCE,
    (
        "Unsupported ICC profile. Convert the photographs to sRGB with a profile-aware editor and upload the converted "
        "copies. If Auto still rejects the exported sRGB profile, select sRGB only for those converted copies. "
        "An input color override does not convert pixels."
    ): _ICC_GUIDANCE,
    "Ambiguous input color; provide input_color='srgb' or 'linear_srgb'": _AMBIGUOUS_GUIDANCE,
    "Ambiguous input color; provide input_color='srgb' or 'linear_srgb' only when known.": _AMBIGUOUS_GUIDANCE,
    (
        "Ambiguous input color. Auto found no usable color-space metadata. Choose Auto with sRGB assumption only if that "
        "assumption is acceptable, select a known source color, or re-export from the original with an embedded profile. "
        "Assumptions are recorded and do not recover the original profile."
    ): (
        "ambiguous_input_color",
        "Input color could not be determined. Auto with sRGB assumption can process untagged photographs if that assumption "
        "is acceptable; it records the assumption without recovering the source profile. Otherwise select the known source "
        "encoding or re-export with an embedded profile, then create a new job.",
    ),
}
_LOG_PREFIXES = ("Stage preprocess failed: ", "RuntimeError: Stage preprocess failed: ")
_MAX_DIAGNOSTIC_LINES = 200
_MAX_DIAGNOSTIC_LINE_CHARS = 1024


def annotate_job_failure(job: ExecutionJob) -> None:
    """Enrich generic failures without returning any untrusted log content.

    The same projection serves new terminal publication and historical status
    reads. It preserves the established error code, exit status and retry policy;
    raw tracebacks and paths never become recovery advice.
    """
    if job.state != "failed" or not isinstance(job.error, dict) or job.error.get("code") != "RUNNER_EXIT_NONZERO":
        return
    pipeline = job.effective_request.get("pipeline") or job.request.get("pipeline")
    if pipeline not in _PHOTOGRAPHY_PIPELINES:
        return

    for line in reversed(job.logs_tail[-_MAX_DIAGNOSTIC_LINES:]):
        if not isinstance(line, str) or len(line) > _MAX_DIAGNOSTIC_LINE_CHARS:
            continue
        for prefix in _LOG_PREFIXES:
            if not line.startswith(prefix):
                continue
            guidance = _FAILURE_GUIDANCE.get(line[len(prefix) :])
            if guidance is None:
                continue
            reason, message = guidance
            original_details = job.error.get("details")
            details = dict(original_details) if isinstance(original_details, Mapping) else {}
            details.update({"stage": "preprocess", "reason": reason})
            job.error = {**job.error, "message": message, "details": details}
            return
