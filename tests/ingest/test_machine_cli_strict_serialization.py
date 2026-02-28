"""Strict machine JSON serialization behavior for metadata CLI."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "test_metadata_extraction.py"



def _load_script_module():
    spec = importlib.util.spec_from_file_location("tp_metadata_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



def test_emit_machine_raises_typed_ingest_error_for_non_finite_payload() -> None:
    module = _load_script_module()
    args = argparse.Namespace(json_pretty=False, json_output=None)
    envelope = {
        "schema": "tp.meta.machine.v1",
        "command": "extract",
        "success": False,
        "exit_code": 5,
        "data": {
            "elapsed_seconds": float("nan"),
        },
        "error": None,
    }

    with pytest.raises(module.OtherIngestFailure) as exc:
        module._emit_machine(envelope, args)

    payload = module._command_error_payload(exc.value)
    assert payload["type"] == "OtherIngestFailure"
    assert payload["exit_code"]["name"] == "OTHER_FAILURE"
    assert payload["exit_code"]["value"] == 5
