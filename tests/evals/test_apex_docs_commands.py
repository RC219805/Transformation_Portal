"""Tests for APEX validation documentation command examples."""

from __future__ import annotations

import importlib.util
import re
import shlex
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
APEX_CLI_MODULES = {
    "tools/audit_apex_assets.py": "audit_apex_assets_docs_unit",
    "tools/run_apex_eval.py": "run_apex_eval_docs_unit",
}
REAL_CANONICAL_RUNBOOK = "APEX_REAL_CANONICAL_EVIDENCE_RUNBOOK.md"
REAL_CANONICAL_ASSET_IDS = {
    "pool_water_stone_001",
    "kitchen_glass_metal_001",
    "exterior_foliage_sky_001",
}


def _load_tool_module(script_path: str, module_name: str):
    path = REPO_ROOT / script_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser_options(script_path: str) -> set[str]:
    module = _load_tool_module(script_path, APEX_CLI_MODULES[script_path])
    parser = module.build_parser()
    return {option for action in parser._actions for option in action.option_strings}


def _documented_apex_commands() -> list[tuple[Path, str, set[str]]]:
    commands: list[tuple[Path, str, set[str]]] = []
    for doc_path in sorted((REPO_ROOT / "docs" / "validation").glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        for block in re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL):
            logical_lines = block.replace("\\\n", " ").splitlines()
            for raw_line in logical_lines:
                line = raw_line.strip()
                if not line:
                    continue
                tokens = shlex.split(line)
                for script_path in APEX_CLI_MODULES:
                    if script_path not in tokens:
                        continue
                    script_index = tokens.index(script_path)
                    options = {token.split("=", 1)[0] for token in tokens[script_index + 1 :] if token.startswith("--")}
                    commands.append((doc_path, script_path, options))
    return commands


def _commands_for_doc(doc_name: str, script_path: str) -> list[set[str]]:
    return [
        options
        for doc_path, documented_script_path, options in _documented_apex_commands()
        if doc_path.name == doc_name and documented_script_path == script_path
    ]


def test_validation_docs_apex_commands_use_current_cli_options():
    known_options = {script_path: _parser_options(script_path) for script_path in APEX_CLI_MODULES}
    failures = []

    for doc_path, script_path, documented_options in _documented_apex_commands():
        unknown_options = sorted(documented_options - known_options[script_path])
        if unknown_options:
            failures.append(f"{doc_path.relative_to(REPO_ROOT)} {script_path}: {', '.join(unknown_options)}")

    assert failures == []


def test_real_canonical_runbook_audit_command_uses_current_cli_options():
    script_path = "tools/audit_apex_assets.py"
    known_options = _parser_options(script_path)
    commands = _commands_for_doc(REAL_CANONICAL_RUNBOOK, script_path)

    assert commands
    assert all(options <= known_options for options in commands)
    assert any("--require-canonical" in options for options in commands)


def test_real_canonical_runbook_evidence_command_uses_current_cli_options():
    script_path = "tools/run_apex_eval.py"
    known_options = _parser_options(script_path)
    commands = _commands_for_doc(REAL_CANONICAL_RUNBOOK, script_path)

    assert commands
    assert all(options <= known_options for options in commands)


def test_real_canonical_runbook_evidence_command_documents_required_flags():
    required = {
        "--asset-root",
        "--candidate-output",
        "--candidate-evidence",
        "--run-scope-asset-id",
        "--emit-evidence-bundle",
        "--synthetic-data",
    }
    commands = _commands_for_doc(REAL_CANONICAL_RUNBOOK, "tools/run_apex_eval.py")

    assert any(required.issubset(options) for options in commands)


def test_real_canonical_runbook_full_manifest_command_mentions_all_three_asset_ids():
    text = (REPO_ROOT / "docs" / "validation" / REAL_CANONICAL_RUNBOOK).read_text(encoding="utf-8")

    assert REAL_CANONICAL_ASSET_IDS <= set(re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", text))


def test_evidence_bundle_command_example_documents_run_scope_and_synthetic_mode():
    commands = [
        options
        for doc_path, script_path, options in _documented_apex_commands()
        if doc_path.name == "APEX_VISUAL_QUALITY_PROTOCOL.md"
        and script_path == "tools/run_apex_eval.py"
        and "--emit-evidence-bundle" in options
    ]

    assert commands
    assert any({"--run-scope-asset-id", "--synthetic-data"}.issubset(options) for options in commands)
