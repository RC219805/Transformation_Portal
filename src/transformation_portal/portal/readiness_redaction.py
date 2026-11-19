"""Redaction of host details from the pipeline readiness matrix.

``GET /v1/readiness`` is reachable without an API key so that probes and the
managed front door can read execution truth. The per-pipeline payloads built
by ``app.py`` also carry host details meant for operators: the absolute path
of the server interpreter, the archive-governance script path, the full
runner argv and the resolved DA3 runtime path. Callers that do not present a
valid API key get the same matrix with those details removed.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

# runner_details keys that hold absolute host paths or argv built from them.
HOST_RUNNER_DETAIL_KEYS = ("python_executable", "script_path", "command")

# Prefix of the operator note that embeds the resolved DA3 runtime path.
DA3_CANARY_RUNTIME_NOTE_PREFIX = "Repo-local DA3 canary runtime resolved at "
_DA3_CANARY_RUNTIME_NOTE_REDACTED = "Repo-local DA3 canary runtime resolved."


def redact_pipeline_readiness_host_details(pipelines: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy of the readiness matrix without host paths.

    Status, prerequisites, canonical commands and every other field are kept
    as they are; only the host-path runner details and path-bearing notes are
    removed or rewritten.
    """

    redacted: Dict[str, Any] = {}
    for name, payload in pipelines.items():
        if not isinstance(payload, Mapping):
            redacted[name] = payload
            continue
        item = dict(payload)
        runner_details = item.get("runner_details")
        if isinstance(runner_details, Mapping):
            item["runner_details"] = {
                key: value for key, value in runner_details.items() if key not in HOST_RUNNER_DETAIL_KEYS
            }
        notes = item.get("notes")
        if isinstance(notes, list):
            item["notes"] = [
                (
                    _DA3_CANARY_RUNTIME_NOTE_REDACTED
                    if isinstance(note, str) and note.startswith(DA3_CANARY_RUNTIME_NOTE_PREFIX)
                    else note
                )
                for note in notes
            ]
        redacted[name] = item
    return redacted


__all__ = [
    "DA3_CANARY_RUNTIME_NOTE_PREFIX",
    "HOST_RUNNER_DETAIL_KEYS",
    "redact_pipeline_readiness_host_details",
]
