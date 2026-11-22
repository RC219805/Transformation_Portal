"""Unit tests for the readiness matrix host-path redaction helper."""

from __future__ import annotations

import pytest

from transformation_portal.portal.readiness_redaction import (
    DA3_CANARY_RUNTIME_NOTE_PREFIX,
    redact_pipeline_readiness_host_details,
)

pytestmark = [pytest.mark.unit, pytest.mark.security]


def test_redaction_drops_host_runner_details_and_rewrites_runtime_note():
    pipelines = {
        "archive-gate-a": {
            "status": "degraded",
            "canonical_command": "fixity-scan",
            "missing_prerequisites": [{"reason": "archive_index_required"}],
            "runner_details": {
                "type": "python_script",
                "available": True,
                "script_path": "/srv/tp/tools/archive_governance.py",
                "python_executable": "/srv/tp/.venv/bin/python",
                "command": ["/srv/tp/.venv/bin/python", "/srv/tp/tools/archive_governance.py", "--json", "fixity-scan"],
            },
            "notes": ["existing archive index required"],
        },
        "lux-depth-v3": {
            "status": "ready",
            "runner_details": {"type": "python_module", "module": "transformation_portal.lux_depth_v3"},
            "notes": [f"{DA3_CANARY_RUNTIME_NOTE_PREFIX}/srv/tp/.runtime/da3/bin/python.", "other note"],
        },
        "lux-depth-v6": {"status": "blocked", "runner_details": {"adapter": "tp.job.photography.bindings.v1"}},
    }

    redacted = redact_pipeline_readiness_host_details(pipelines)

    assert redacted["archive-gate-a"]["runner_details"] == {"type": "python_script", "available": True}
    assert redacted["archive-gate-a"]["missing_prerequisites"] == [{"reason": "archive_index_required"}]
    assert redacted["archive-gate-a"]["notes"] == ["existing archive index required"]
    assert redacted["lux-depth-v3"]["notes"] == ["Repo-local DA3 canary runtime resolved.", "other note"]
    assert redacted["lux-depth-v6"] == pipelines["lux-depth-v6"]
    assert "/srv/tp" not in repr(redacted)
    # The source matrix is left untouched for authenticated callers.
    assert pipelines["archive-gate-a"]["runner_details"]["script_path"] == "/srv/tp/tools/archive_governance.py"
