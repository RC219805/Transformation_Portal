"""Unit tests for ``scripts/validation/generate_design_tokens_doc.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validation" / "generate_design_tokens_doc.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_design_tokens_doc", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_css_variables_extracts_root_block_declarations() -> None:
    module = _load_module()
    css = """
    :root {
        --ux-body-size: 1rem;
        --ux-text-primary: #1e293b;
    }
    """
    decls = module.parse_token_declarations(css, "fixture.css")

    by_name = {d.name: d for d in decls}
    assert by_name["--ux-body-size"].value == "1rem"
    assert by_name["--ux-body-size"].scope == module.SCOPE_LIGHT
    assert by_name["--ux-text-primary"].value == "#1e293b"
    assert by_name["--ux-text-primary"].source == "fixture.css"


def test_parse_css_variables_separates_dark_mode_overrides() -> None:
    module = _load_module()
    css = """
    :root {
        --ux-text-primary: #1e293b;
        --ux-focus-ring: rgba(8, 145, 178, 0.92);
    }

    :root.dark,
    .dark:root {
        --ux-text-primary: #e2e8f0;
        --ux-focus-ring: rgba(103, 232, 249, 0.96);
    }
    """
    decls = module.parse_token_declarations(css, "fixture.css")

    light = [d for d in decls if d.scope == module.SCOPE_LIGHT]
    dark = [d for d in decls if d.scope == module.SCOPE_DARK]
    assert {d.name for d in light} == {"--ux-text-primary", "--ux-focus-ring"}
    assert {d.name for d in dark} == {"--ux-text-primary", "--ux-focus-ring"}
    dark_by_name = {d.name: d.value for d in dark}
    assert dark_by_name["--ux-text-primary"] == "#e2e8f0"
    assert dark_by_name["--ux-focus-ring"] == "rgba(103, 232, 249, 0.96)"


def test_parse_css_variables_captures_reduced_motion_overrides() -> None:
    module = _load_module()
    css = """
    :root {
        --ux-motion-fast: 160ms;
        --ux-motion-normal: 220ms;
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
        }

        :root {
            --ux-motion-fast: 0ms;
            --ux-motion-normal: 0ms;
        }
    }
    """
    decls = module.parse_token_declarations(css, "fixture.css")

    reduced = [d for d in decls if d.scope == module.SCOPE_REDUCED_MOTION]
    assert {d.name for d in reduced} == {"--ux-motion-fast", "--ux-motion-normal"}
    assert all(d.value == "0ms" for d in reduced)


def test_derive_description_handles_known_token_stems() -> None:
    module = _load_module()
    cases = {
        "--ux-text-primary": "Primary text color",
        "--ux-space-4": "Spacing scale step 4",
        "--ux-radius-pill": "Pill radius",
        "--ux-motion-fast": "Fast motion duration",
        "--ux-status-warning": "Warning status color",
        "--shell-veil-soft": "Soft veil overlay",
        "--shell-accent-fill-strong": "Strong accent fill color",
        "--ambient-color-a": "Ambient color A",
        "--ambient-stage-rotate": "Ambient stage rotation",
    }
    for name, expected in cases.items():
        assert module.derive_description(name) == expected, name


def test_derive_description_falls_back_to_humanized_name_for_unknown_stems() -> None:
    module = _load_module()
    assert module.derive_description("--ux-mystery-knob") == "Ux mystery knob"
    assert module.derive_description("--shell-experimental-glow") == "Shell experimental glow"


def test_render_markdown_groups_tokens_by_namespace_prefix() -> None:
    module = _load_module()
    entries = [
        module.TokenEntry(name="--ux-body-size", light="1rem"),
        module.TokenEntry(name="--ux-panel-border", light="rgba(0,0,0,0.1)"),
        module.TokenEntry(name="--shell-ink", light="#000", dark="#fff"),
        module.TokenEntry(name="--ambient-stage-scale", light="1"),
    ]
    rendered = module.render_document(entries)

    ux_idx = rendered.index("## Shared UI tokens (`--ux-*`)")
    panel_idx = rendered.index("## Panel tokens (`--ux-panel-*`)")
    shell_idx = rendered.index("## Shell tokens (`--shell-*`)")
    ambient_idx = rendered.index("## Ambient tokens (`--ambient-*`)")

    assert ux_idx < panel_idx < shell_idx < ambient_idx
    assert "`--ux-body-size`" in rendered
    assert "`--ux-panel-border`" in rendered
    assert "`--shell-ink`" in rendered
    assert "`--ambient-stage-scale`" in rendered
    # Panel tokens must not bleed into the shared-UI section.
    ux_section = rendered[ux_idx:panel_idx]
    assert "--ux-panel-border" not in ux_section


def test_check_mode_returns_zero_when_committed_doc_matches_sources(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"--check failed: stdout={result.stdout!r}, stderr={result.stderr!r}"
    assert "up to date" in result.stdout


def test_check_mode_returns_nonzero_with_diff_on_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    doc_path = tmp_path / "tokens.md"
    committed = module.GENERATED_DOC_PATH.read_text(encoding="utf-8")
    doc_path.write_text(committed + "\n<!-- drift -->\n", encoding="utf-8")
    monkeypatch.setattr(module, "GENERATED_DOC_PATH", doc_path)

    result = module.main(["--check"])
    captured = capsys.readouterr()

    assert result != 0
    assert "drifted" in captured.err
    assert module.REGENERATE_COMMAND in captured.err
