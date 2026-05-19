"""Contract checks for Docker runtime image construction."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"
pytestmark = pytest.mark.unit


def _dockerfile_text() -> str:
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


def _dockerfile_stages() -> dict[str, str]:
    stages: dict[str, list[str]] = {}
    current_stage: str | None = None

    for line in _dockerfile_text().splitlines():
        match = re.match(r"^FROM\s+\S+(?:\s+as\s+([^\s]+))?", line, flags=re.IGNORECASE)
        if match:
            current_stage = match.group(1) or f"unnamed-{len(stages)}"
            stages[current_stage] = [line]
            continue
        if current_stage is not None:
            stages[current_stage].append(line)

    return {stage: "\n".join(lines) for stage, lines in stages.items()}


def test_python_slim_base_is_codename_pinned() -> None:
    dockerfile = _dockerfile_text()

    assert re.search(r"^FROM python:3\.11-slim-trixie AS python-runtime-base$", dockerfile, flags=re.MULTILINE)
    assert "FROM python:3.11-slim AS " not in dockerfile


def test_runtime_stages_do_not_install_compiler_tooling() -> None:
    stages = _dockerfile_stages()
    runtime_stages = ("python-runtime-base", "cpu", "gpu-runtime-base", "gpu", "apple-silicon")
    compiler_packages = ("build-essential", "python3.11-dev", "python3-pip")

    for stage in runtime_stages:
        stage_text = stages[stage]
        for package in compiler_packages:
            assert package not in stage_text, f"{package} leaked into runtime stage {stage}"

    assert "build-essential" in stages["python-build"]
    assert "build-essential" in stages["gpu-build"]


def test_gpu_image_invokes_python_311_explicitly() -> None:
    stages = _dockerfile_stages()
    gpu_build = stages["gpu-build"]
    gpu_runtime = stages["gpu"]
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "pip3" not in gpu_build
    assert "python3.11 -m pip install" in gpu_build
    assert "CMD python3.11 -c" in gpu_runtime
    assert 'CMD ["python3.11", "-m", "transformation_portal.cli"]' in gpu_runtime
    assert re.search(r"(?<![\w.])python3(?!\.11|\w)", gpu_runtime) is None

    assert "command: python3.11 -m transformation_portal.cli serve --host 0.0.0.0 --port 8000" in compose
    assert "command: python3 -m transformation_portal.cli serve --host 0.0.0.0 --port 8000" not in compose


def test_dockerignore_excludes_heavy_local_context() -> None:
    ignored_entries = {
        line.strip().rstrip("/")
        for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert {
        ".git",
        ".venv",
        ".runtime",
        "node_modules",
        "output",
        "input_images",
        "checkpoints",
        ".env",
        ".env.*",
        "!.env.example",
        ".npmrc",
        ".pypirc",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "*.safetensors",
        "*.pt",
        "*.pth",
    } <= ignored_entries
