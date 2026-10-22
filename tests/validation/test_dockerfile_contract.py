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


def test_cpu_source_and_runtime_share_a_clean_codename_pinned_base() -> None:
    """Do not merge a clean prefix over upstream's vulnerable bootstrap files."""
    stages = _dockerfile_stages()
    builder = stages["cpu-python-build"]
    runtime = stages["python-runtime-base"]
    base_pattern = r"^FROM (debian:trixie-slim@sha256:[a-f0-9]{64}) AS "

    builder_base = re.match(base_pattern, builder)
    runtime_base = re.match(base_pattern, runtime)
    assert builder_base is not None and runtime_base is not None
    assert builder_base.group(1) == runtime_base.group(1)
    assert "COPY --from=cpu-python-build /opt/python3.11-runtime/ /opt/python3.11/" in runtime
    assert "COPY --from=gpu-python-build" not in runtime
    assert "sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == 'final'" in runtime


def test_cpu_preserves_python_and_pip_launchers() -> None:
    stages = _dockerfile_stages()
    builder = stages["cpu-python-build"]

    assert "ln -s python3.11 /opt/python3.11/bin/python3" in builder
    assert "ln -s python3 /opt/python3.11/bin/python" in builder
    assert "ln -s pip3.11 /opt/python3.11/bin/pip3" in builder
    assert "ln -s pip3 /opt/python3.11/bin/pip" in builder
    assert 'ENV PATH="/opt/python3.11/bin:${PATH}"' in stages["python-runtime-base"]


@pytest.mark.parametrize(
    ("builder_stage", "runtime_stage"),
    (("cpu-python-build", "python-runtime-base"), ("gpu-python-build", "gpu-runtime-base")),
)
def test_runtimes_preserve_loadable_sqlite_extension_support(builder_stage: str, runtime_stage: str) -> None:
    stages = _dockerfile_stages()

    assert "--enable-loadable-sqlite-extensions" in stages[builder_stage]
    assert "assert hasattr(sqlite3.Connection, 'enable_load_extension')" in stages[runtime_stage]
    assert "sqlite3.connect(':memory:').enable_load_extension(False)" in stages[runtime_stage]


def test_runtime_stages_do_not_install_compiler_tooling() -> None:
    stages = _dockerfile_stages()
    runtime_stages = ("python-runtime-base", "cpu", "gpu-runtime-base", "gpu", "apple-silicon")
    compiler_packages = ("build-essential", "python3.11-dev", "python3-pip")

    for stage in runtime_stages:
        stage_text = stages[stage]
        for package in compiler_packages:
            assert package not in stage_text, f"{package} leaked into runtime stage {stage}"

    assert "build-essential" in stages["python-build"]
    assert "build-essential" in stages["cpu-python-build"]
    assert "build-essential" in stages["gpu-python-build"]
    assert "build-essential" in stages["gpu-build"]


@pytest.mark.parametrize("builder_stage", ("cpu-python-build", "gpu-python-build"))
def test_python_builders_use_verified_final_source(builder_stage: str) -> None:
    stages = _dockerfile_stages()
    builder = stages[builder_stage]

    assert re.search(r"^ENV PYTHON_VERSION=3\.11\.\d+$", builder, flags=re.MULTILINE)
    assert re.search(r"^ENV PYTHON_SHA256=[a-f0-9]{64}$", builder, flags=re.MULTILINE)
    assert '"https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz"' in builder
    source_checksum = 'echo "${PYTHON_SHA256}  /tmp/python.tar.xz" | sha256sum --check --strict'
    assert builder.index(source_checksum) < builder.index("tar --extract") < builder.index("./configure")
    assert "--with-ensurepip=install" in builder
    assert "sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == 'final'" in builder


def test_gpu_python_uses_the_runtime_abi_and_rejects_prereleases() -> None:
    """Jammy's prerelease Python must not satisfy the final-3.11 contract."""
    stages = _dockerfile_stages()
    builder = stages["gpu-python-build"]
    runtime = stages["gpu-runtime-base"]

    assert builder.startswith("FROM ubuntu:22.04 AS gpu-python-build")
    assert re.search(
        r"^FROM nvidia/cuda:12\.6\.3-runtime-ubuntu22\.04@sha256:[a-f0-9]{64} AS gpu-runtime-base$",
        runtime,
        flags=re.MULTILINE,
    )
    assert "COPY --from=gpu-python-build" in runtime
    assert "python3.11-venv" not in runtime
    assert not re.search(r"^\s+python3\.11(?:\s|\\|$)", runtime, flags=re.MULTILINE)
    assert "add-apt-repository" not in builder + runtime
    for stage in (builder, runtime):
        assert "sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == 'final'" in stage


@pytest.mark.parametrize("builder_stage", ("cpu-python-build", "gpu-python-build"))
def test_venv_bootstrap_uses_verified_patched_wheels(builder_stage: str) -> None:
    """Upgrading a venv must not leave vulnerable bundled bootstrap wheels."""
    builder = _dockerfile_stages()[builder_stage]

    assert "--with-wheel-pkg-dir=/opt/python3.11/share/python-wheels" in builder
    assert "https://files.pythonhosted.org/packages/" in builder
    for wheel in ("pip-26.2.2-py3-none-any.whl", "setuptools-83.0.0-py3-none-any.whl"):
        checksum = re.search(
            rf'echo "[a-f0-9]{{64}}  /opt/python3\.11/share/python-wheels/{re.escape(wheel)}" \| sha256sum --check --strict',
            builder,
        )
        assert checksum is not None
        assert checksum.start() < builder.index("./configure")
    assert builder.index("rm -rf /opt/python3.11/lib/python3.11/ensurepip/_bundled") < builder.index(
        "cp -a /opt/python3.11 /opt/python3.11-runtime"
    )


def test_cpu_and_gpu_security_source_pins_stay_aligned() -> None:
    stages = _dockerfile_stages()
    cpu_builder = stages["cpu-python-build"]
    gpu_builder = stages["gpu-python-build"]
    python_pins = r"^ENV PYTHON_(?:VERSION|SHA256)=.+$"
    wheel_hashes = r'"([a-f0-9]{64})  /opt/python3\.11/share/python-wheels/([^"/]+)"'

    assert re.findall(python_pins, cpu_builder, flags=re.MULTILINE) == re.findall(python_pins, gpu_builder, flags=re.MULTILINE)
    assert re.findall(wheel_hashes, cpu_builder) == re.findall(wheel_hashes, gpu_builder)


def test_cpu_application_builder_uses_matching_headers_and_pinned_tools() -> None:
    builder = _dockerfile_stages()["python-build"]

    assert "COPY --from=cpu-python-build /opt/python3.11/include/ /opt/python3.11/include/" in builder
    assert "python -m venv ${VIRTUAL_ENV}" in builder
    assert "pip==26.2.2 setuptools==83.0.0 wheel==0.48.0" in builder


def test_gpu_build_uses_matching_python_headers_and_patched_cuda_wheels() -> None:
    stages = _dockerfile_stages()
    builder = stages["gpu-build"]

    assert "COPY --from=gpu-python-build /opt/python3.11/include/ /opt/python3.11/include/" in builder
    assert "python3.11-dev" not in builder
    assert "python3-pip" not in builder
    assert "python3.11 -m venv ${VIRTUAL_ENV}" in builder
    assert "pip==26.2.2 setuptools==83.0.0 wheel==0.48.0" in builder
    assert "torch==2.13.0+cu126 torchvision==0.28.0+cu126" in builder
    assert "--index-url https://download.pytorch.org/whl/cu126" in builder
    assert "cu121" not in builder


def test_runtime_images_invoke_python_311_explicitly() -> None:
    stages = _dockerfile_stages()
    cpu_runtime = stages["cpu"]
    gpu_build = stages["gpu-build"]
    gpu_runtime = stages["gpu"]
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "pip3" not in gpu_build
    assert "python3.11 -m pip install" in gpu_build
    assert "CMD python3.11 -c" in cpu_runtime
    assert 'CMD ["python3.11", "-m", "transformation_portal.cli"]' in cpu_runtime
    assert "CMD python3.11 -c" in gpu_runtime
    assert 'CMD ["python3.11", "-m", "transformation_portal.cli"]' in gpu_runtime
    assert 'CMD ["python", "-m", "transformation_portal.cli"]' not in cpu_runtime
    assert re.search(r"(?<![\w.])python3(?!\.11|\w)", gpu_runtime) is None

    assert "command: python3.11 -m transformation_portal.cli serve --host 0.0.0.0 --port 8000" in compose
    assert (
        "command: python3.11 -m transformation_portal.cli batch-process " "/app/input /app/output --preset golden_hour"
    ) in compose
    assert "command: python3.11 -m transformation_portal.monitoring.dashboard --port 8080" in compose
    assert "        - python3.11" in compose
    assert "command: python -m transformation_portal" not in compose
    assert "command: python3 -m transformation_portal.cli serve --host 0.0.0.0 --port 8000" not in compose


def test_compose_mounts_governed_local_input_surface() -> None:
    """Compose should not create untracked root input directories."""
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "./input_images:/app/input:ro" in compose
    assert "./input:/app/input:ro" not in compose


def test_docker_targets_do_not_bypass_governed_ml_lock_contract() -> None:
    """Docker targets must not install broad or platform-wrong ML extras."""
    dockerfile = _dockerfile_text()
    stages = _dockerfile_stages()
    apple_silicon_stage = stages["apple-silicon"]

    assert 'pip install --no-cache-dir -e ".[ml]"' not in dockerfile
    assert "pip install --no-cache-dir coremltools" not in dockerfile
    assert re.search(r"^FROM cpu AS apple-silicon$", dockerfile, flags=re.MULTILINE)
    assert "Docker on Apple Silicon still runs a Linux container" in dockerfile
    assert "make install-ml-core" in dockerfile
    assert "ENV DEVICE=cpu" in apple_silicon_stage
    assert "ENV DEVICE=mps" not in apple_silicon_stage


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
        ".pyre",
        ".coverage",
        ".coverage.*",
        "node_modules",
        "coverage.xml",
        "coverage.json",
        "htmlcov",
        ".rag_cache",
        "build",
        "dist",
        "downloads",
        "docs/api/_build",
        "output",
        "outputs",
        "processed_images",
        "processed_output",
        "extracted_context",
        "artifacts",
        "archive_reports",
        "local_artifacts",
        "depth_maps_apex",
        "artifact_store",
        "archive/experiments",
        "archive/deprecated",
        "archive/legacy",
        "input_images",
        "data/sample_images",
        "Architectural_Plans",
        "checkpoints",
        "models",
        "weights",
        "pretrained",
        ".env",
        ".env.*",
        "!.env.example",
        "external",
        ".cas",
        ".local_backup",
        ".backup_local",
        ".branch_cleanup_backup",
        ".npmrc",
        ".pypirc",
        "web/secure-landing/.metafiles",
        "web/secure-landing/tmp",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "*.safetensors",
        "*.pt",
        "*.pth",
        "*.xlsx",
        "*.xls",
        "tmp_tensor.npy",
        "uv.lock",
    } <= ignored_entries
