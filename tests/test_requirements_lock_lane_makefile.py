from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE_PATH = PROJECT_ROOT / "requirements" / "Makefile"
GENERIC_LOCK_FILES = ("all.txt", "base.txt", "dev.txt", "ci.txt", "security.txt", "tools-archive.txt")


def _read_makefile() -> str:
    return MAKEFILE_PATH.read_text(encoding="utf-8")


def _target_body(name: str) -> str:
    text = _read_makefile()
    match = re.search(rf"^{re.escape(name)}:(?:[^\n]*)\n(?P<body>(?:\t.*\n)+)", text, flags=re.MULTILINE)
    assert match is not None, f"Makefile target {name} not found"
    return match.group("body")


def test_generic_aggregate_owns_pip_tools_click_compatibility_bound() -> None:
    all_input = (PROJECT_ROOT / "requirements" / "all.in").read_text(encoding="utf-8")
    base_input = (PROJECT_ROOT / "requirements" / "base.in").read_text(encoding="utf-8")

    assert "click>=8.4.2,<8.5" in all_input
    assert not any(line.lstrip().startswith("click") for line in base_input.splitlines())


def test_generic_targets_do_not_reference_target_owned_ml_locks() -> None:
    text = _read_makefile()
    assert "compile: compile-generic" in text
    assert "update: update-generic" in text
    assert "check: check-generic" in text

    for target in ("compile-generic", "update-generic", "check-generic"):
        body = _target_body(target)
        assert "ml-core-darwin-arm64.txt" not in body
        assert "ml-core-darwin-x86_64.txt" not in body
        assert "ml-core-linux.txt" not in body


def test_update_generic_repairs_staged_marker_constraints_before_dependents() -> None:
    body = _target_body("_prepare-generic-lock-set")

    compile_all = body.index("$(PIP_COMPILE) $(GENERIC_PIP_COMPILE_ARGS) all.in -o all.txt")
    repair_all = body.index('--lockfile "$$tmp_dir/all.txt"')
    compile_base = body.index("$(PIP_COMPILE) $(GENERIC_PIP_COMPILE_ARGS) -c all.txt base.in -o base.txt")
    repair_base = body.index('--lockfile "$$tmp_dir/base.txt"')
    compile_dev = body.index("$(PIP_COMPILE) $(GENERIC_PIP_COMPILE_ARGS) -c all.txt dev.in -o dev.txt")

    assert compile_all < repair_all < compile_base < repair_base < compile_dev


def test_all_public_generic_writers_delegate_to_one_transaction() -> None:
    text = _read_makefile()

    assert "$(GENERIC_LOCK_FILES): compile-generic" in text
    assert "_publish-generic-lock-set:\n" in text
    assert "_publish-generic-lock-set: require-pip-compile" not in text
    assert "_publish-generic-lock-set GENERIC_PIP_COMPILE_ARGS=" in _target_body("compile-generic")
    assert "_publish-generic-lock-set GENERIC_PIP_COMPILE_ARGS=--upgrade" in _target_body("update-generic")
    assert "GENERIC_LOCK_SEED_ARGS=--seed-existing" in _target_body("compile-generic")
    assert "GENERIC_LOCK_SEED_ARGS=" in _target_body("update-generic")
    publisher_body = _target_body("_publish-generic-lock-set")
    preparation_body = _target_body("_prepare-generic-lock-set")
    assert "--prepare-command" in publisher_body
    assert "_prepare-generic-lock-set" in publisher_body
    assert "$(GENERIC_LOCK_SEED_ARGS) --prepare-command" in publisher_body
    assert '--staged-generic-dir "$$tmp_dir"' in preparation_body
    assert (
        publisher_body.index("--recover-only")
        < publisher_body.index("require-pip-compile")
        < publisher_body.index("--prepare-command")
    )


def test_generic_writer_lock_covers_preparation_and_publication() -> None:
    publisher_body = _target_body("_publish-generic-lock-set")
    preparation_body = _target_body("_prepare-generic-lock-set")

    assert "--prepare-command" in publisher_body
    assert "publish_requirement_locks.py" not in preparation_body
    assert "TP_GENERIC_LOCK_STAGING_DIR" in preparation_body


def test_darwin_ml_lanes_require_fresh_generic_base_contract() -> None:
    text = _read_makefile()

    assert re.search(r"^ml-core-darwin-arm64\.txt:.*\bbase\.txt$", text, flags=re.MULTILINE)
    assert re.search(r"^update-ml-darwin-arm64:.*\bbase\.txt$", text, flags=re.MULTILINE)
    assert re.search(r"^check-ml-darwin-arm64:.*\bcheck-generic$", text, flags=re.MULTILINE)


def test_clean_routes_only_generic_locks_through_serialized_cleanup() -> None:
    body = _target_body("clean")

    assert "--clean-generic" in body
    rm_command = next(line for line in body.splitlines() if "rm -f" in line)
    assert all(name not in rm_command for name in ("all.txt", "base.txt", "dev.txt", "ci.txt", "security.txt"))
    assert "tools-archive.txt" not in rm_command
    assert "ml-core-darwin-arm64.txt" in rm_command


def test_check_generic_normalizes_host_specific_generic_lock_packages() -> None:
    body = _target_body("check-generic")

    for snippet in (
        "all.txt:jeepney==*",
        "all.txt:secretstorage==*",
        "all.txt:opencv-python==*",
        "all.txt:opencv-python-headless==*",
        "base.txt:opencv-python==*",
        "base.txt:opencv-python-headless==*",
        "ci.txt:cffi==*",
        "ci.txt:cryptography==*",
        "ci.txt:jeepney==*",
        "ci.txt:pycparser==*",
        "ci.txt:secretstorage==*",
    ):
        assert snippet in body


def _write_fake_pip_compile(path: Path, *, version: str = "7.6.1") -> None:
    path.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import os
import sys

if "--version" in sys.argv:
    print("pip-compile, version __PIP_TOOLS_VERSION__")
    raise SystemExit(0)

if "--help" in sys.argv:
    print("fake pip-compile")
    raise SystemExit(0)

output = None
for index, arg in enumerate(sys.argv):
    if arg == "-o":
        output = sys.argv[index + 1]
        break
    if arg.startswith("--output-file="):
        output = arg.split("=", 1)[1]
        break

if output is None:
    raise SystemExit("missing output path")

path = Path(output)
audit_path = os.environ.get("FAKE_PIP_COMPILE_AUDIT")
if audit_path:
    seed_state = "seeded" if path.exists() else "empty"
    upgrade_state = "upgrade" if "--upgrade" in sys.argv else "conservative"
    with Path(audit_path).open("a", encoding="utf-8") as audit:
        audit.write(f"{path.name}|{seed_state}|{upgrade_state}\\n")
header = [
    "#",
    "# This file is autogenerated by pip-compile with Python 3.11",
    "# by the following command:",
    "#",
    f"#    fake pip-compile --output-file={path.name}",
]

if path.name in {"all.txt", "base.txt"}:
    body = [
        "numpy==9.9.9",
        "opencv-python-headless==4.14.0.92",
        "    # via -r base.in",
        "packaging==99.0",
    ]
elif path.name == "ci.txt":
    body = [
        "keyring==25.7.0",
        "cffi==2.0.0",
        "cryptography==46.0.7",
        "jeepney==0.9.0",
        "pycparser==3.0",
        "secretstorage==3.5.0",
        "packaging==99.0",
    ]
else:
    body = ["packaging==99.0"]

if path.name == "all.txt":
    body.extend(["keyring==25.7.0", "jeepney==0.9.0", "secretstorage==3.5.0"])
if os.environ.get("FAKE_FLOATING_LOCK") == path.name:
    body.append("floating-package>=1.0")

path.write_text("\\n".join(header + body) + "\\n", encoding="utf-8")
""".replace("__PIP_TOOLS_VERSION__", version),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_fake_pip_python(
    path: Path,
    *,
    version: str = "26.2.1",
    click_version: str = "8.4.2",
) -> None:
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then\n'
        f'  echo "pip {version} from /test/site-packages/pip (python 3.11)"\n'
        "  exit 0\n"
        "fi\n"
        'if [ "$1" = "-c" ]; then\n'
        f'  echo "{click_version}"\n'
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.security
def test_require_pip_compile_rejects_stale_toolchain(tmp_path: Path) -> None:
    fake_pip_compile = tmp_path / "pip-compile"
    fake_pip_python = tmp_path / "python"
    _write_fake_pip_compile(fake_pip_compile, version="7.5.2")
    _write_fake_pip_python(fake_pip_python)

    result = subprocess.run(
        [
            "make",
            "require-pip-compile",
            f"PIP_COMPILE_BIN={fake_pip_compile}",
            f"PIP_PYTHON_BIN={fake_pip_python}",
        ],
        cwd=MAKEFILE_PATH.parent,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "requires pip-tools==7.6.1" in output
    assert "pip-compile, version 7.5.2" in output


@pytest.mark.security
def test_require_pip_compile_rejects_stale_pip_runtime(tmp_path: Path) -> None:
    fake_pip_compile = tmp_path / "pip-compile"
    fake_pip_python = tmp_path / "python"
    _write_fake_pip_compile(fake_pip_compile)
    _write_fake_pip_python(fake_pip_python, version="26.1.2")

    result = subprocess.run(
        [
            "make",
            "require-pip-compile",
            f"PIP_COMPILE_BIN={fake_pip_compile}",
            f"PIP_PYTHON_BIN={fake_pip_python}",
        ],
        cwd=MAKEFILE_PATH.parent,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "requires pip==26.2.1" in output
    assert "pip 26.1.2 from /test/site-packages/pip" in output


@pytest.mark.security
def test_require_pip_compile_rejects_incompatible_click_runtime(tmp_path: Path) -> None:
    fake_pip_compile = tmp_path / "pip-compile"
    fake_pip_python = tmp_path / "python"
    _write_fake_pip_compile(fake_pip_compile)
    _write_fake_pip_python(fake_pip_python, click_version="8.5.0")

    result = subprocess.run(
        [
            "make",
            "require-pip-compile",
            f"PIP_COMPILE_BIN={fake_pip_compile}",
            f"PIP_PYTHON_BIN={fake_pip_python}",
        ],
        cwd=MAKEFILE_PATH.parent,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "requires click==8.4.2 for pip-tools command provenance" in output
    assert "reported Click '8.5.0'" in output


@pytest.mark.security
def test_require_pip_compile_accepts_exact_governed_toolchain(tmp_path: Path) -> None:
    fake_pip_compile = tmp_path / "pip-compile"
    fake_pip_python = tmp_path / "python"
    _write_fake_pip_compile(fake_pip_compile)
    _write_fake_pip_python(fake_pip_python)

    result = subprocess.run(
        [
            "make",
            "require-pip-compile",
            f"PIP_COMPILE_BIN={fake_pip_compile}",
            f"PIP_PYTHON_BIN={fake_pip_python}",
        ],
        cwd=MAKEFILE_PATH.parent,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def _write_lock_with_opencv_marker_pins(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#",
                "# This file is autogenerated by pip-compile with Python 3.11",
                "# by the following command:",
                "#",
                f"#    pip-compile --output-file={path.name}",
                "numpy==2.4.4",
                'opencv-python==4.13.0.92 ; platform_system != "Linux"',
                "    # via -r base.in",
                'opencv-python-headless==4.13.0.92 ; platform_system == "Linux"',
                "    # via -r base.in",
                "packaging==26.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _create_isolated_lock_lane(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    requirements_dir = repo_root / "requirements"
    utilities_dir = repo_root / "scripts" / "utilities"
    validation_dir = repo_root / "scripts" / "validation"
    requirements_dir.mkdir(parents=True)
    utilities_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    (requirements_dir / "Makefile").write_text(MAKEFILE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    for script_name in ("restore_requirement_marker_pins.py", "publish_requirement_locks.py"):
        (utilities_dir / script_name).write_text(
            (PROJECT_ROOT / "scripts" / "utilities" / script_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    for script_name in (
        "check_dependency_pinning.py",
        "check_lock_ownership.py",
        "check_requirements_lock_contract.py",
    ):
        (validation_dir / script_name).write_text(
            (PROJECT_ROOT / "scripts" / "validation" / script_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    for name in ("all.in", "base.in", "dev.in", "ci.in", "security.in", "tools-archive.in"):
        (requirements_dir / name).write_text("# test input\n", encoding="utf-8")
    (requirements_dir / "base.in").write_text(
        'opencv-python>=4.8.0,<5 ; platform_system != "Linux"\n'
        'opencv-python-headless>=4.8.0,<5 ; platform_system == "Linux"\n',
        encoding="utf-8",
    )
    return requirements_dir


def _write_complete_previous_lock_set(requirements_dir: Path) -> dict[str, bytes]:
    for name in ("all.txt", "base.txt"):
        _write_lock_with_opencv_marker_pins(requirements_dir / name)
    for name in ("dev.txt", "ci.txt", "security.txt", "tools-archive.txt"):
        (requirements_dir / name).write_text(
            "#\n"
            "# This file is autogenerated by pip-compile with Python 3.11\n"
            "# by the following command:\n"
            "#\n"
            f"#    pip-compile --output-file={name}\n"
            f"example-{name.removesuffix('.txt')}==1.0\n",
            encoding="utf-8",
        )
    return {name: (requirements_dir / name).read_bytes() for name in GENERIC_LOCK_FILES}


def _write_stale_mixed_publication(
    requirements_dir: Path,
    original: dict[str, bytes],
) -> Path:
    transaction_dir = requirements_dir / ".generic-lock-publish-stale-make"
    backups_dir = transaction_dir / "backups"
    backups_dir.mkdir(parents=True)
    for name, content in original.items():
        (backups_dir / name).write_bytes(content)

    touched = list(GENERIC_LOCK_FILES[:2])
    for name in touched:
        (requirements_dir / name).write_text(f"mixed-{name}==2.0\n", encoding="utf-8")
    (transaction_dir / "journal.json").write_text(
        json.dumps(
            {
                "version": 1,
                "destination": str(requirements_dir.resolve()),
                "names": list(GENERIC_LOCK_FILES),
                "existing": list(GENERIC_LOCK_FILES),
                "touched": touched,
                "state": "publishing",
            }
        ),
        encoding="utf-8",
    )
    return transaction_dir


@pytest.mark.parametrize(
    ("target", "seed_previous"),
    [
        ("compile-generic", True),
        ("update-generic", True),
        ("base.txt", True),
        ("compile-generic", False),
    ],
)
def test_generic_writer_preserves_marker_pins_and_publishes_complete_set(
    tmp_path: Path,
    target: str,
    seed_previous: bool,
) -> None:
    requirements_dir = _create_isolated_lock_lane(tmp_path)
    if seed_previous:
        for name in ("all.txt", "base.txt"):
            _write_lock_with_opencv_marker_pins(requirements_dir / name)
    ml_lock = requirements_dir / "ml-core-darwin-arm64.txt"
    ml_lock.write_text("torch==2.13.0\n", encoding="utf-8")

    fake_pip_compile = tmp_path / "pip-compile"
    fake_pip_python = tmp_path / "python"
    _write_fake_pip_compile(fake_pip_compile)
    _write_fake_pip_python(fake_pip_python)

    result = subprocess.run(
        [
            "make",
            target,
            f"PIP_COMPILE_BIN={fake_pip_compile}",
            f"PIP_PYTHON_BIN={fake_pip_python}",
        ],
        cwd=requirements_dir,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("all.txt", "base.txt"):
        text = (requirements_dir / name).read_text(encoding="utf-8")
        assert 'opencv-python==4.14.0.92 ; platform_system != "Linux"' in text
        assert 'opencv-python-headless==4.14.0.92 ; platform_system == "Linux"' in text
        if seed_previous:
            assert "opencv-python==4.13.0.92" not in text
            assert "opencv-python-headless==4.13.0.92" not in text
    for name in ("all.txt", "base.txt", "dev.txt", "ci.txt", "security.txt", "tools-archive.txt"):
        text = (requirements_dir / name).read_text(encoding="utf-8")
        assert f"--output-file={name}" in text
        assert str(tmp_path) not in text
    assert ml_lock.read_text(encoding="utf-8") == "torch==2.13.0\n"


@pytest.mark.parametrize(
    ("target", "expected_seed", "expected_mode"),
    [
        ("compile-generic", "seeded", "conservative"),
        ("update-generic", "empty", "upgrade"),
    ],
)
def test_generic_writer_seeds_only_conservative_compiles(
    tmp_path: Path,
    target: str,
    expected_seed: str,
    expected_mode: str,
) -> None:
    requirements_dir = _create_isolated_lock_lane(tmp_path)
    _write_complete_previous_lock_set(requirements_dir)
    fake_pip_compile = tmp_path / "pip-compile"
    fake_pip_python = tmp_path / "python"
    _write_fake_pip_compile(fake_pip_compile)
    _write_fake_pip_python(fake_pip_python)
    audit_path = tmp_path / "pip-compile-audit.txt"

    result = subprocess.run(
        [
            "make",
            target,
            f"PIP_COMPILE_BIN={fake_pip_compile}",
            f"PIP_PYTHON_BIN={fake_pip_python}",
        ],
        cwd=requirements_dir,
        env={**os.environ, "FAKE_PIP_COMPILE_AUDIT": str(audit_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert audit_path.read_text(encoding="utf-8").splitlines() == [
        f"{name}|{expected_seed}|{expected_mode}" for name in GENERIC_LOCK_FILES
    ]


@pytest.mark.security
def test_generic_writer_recovers_stale_set_before_toolchain_validation_failure(tmp_path: Path) -> None:
    requirements_dir = _create_isolated_lock_lane(tmp_path)
    original = _write_complete_previous_lock_set(requirements_dir)
    transaction_dir = _write_stale_mixed_publication(requirements_dir, original)
    fake_pip_compile = tmp_path / "pip-compile"
    fake_pip_python = tmp_path / "python"
    _write_fake_pip_compile(fake_pip_compile, version="7.5.2")
    _write_fake_pip_python(fake_pip_python)

    result = subprocess.run(
        [
            "make",
            "compile-generic",
            f"PIP_COMPILE_BIN={fake_pip_compile}",
            f"PIP_PYTHON_BIN={fake_pip_python}",
        ],
        cwd=requirements_dir,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "generic requirements lock recovery completed successfully" in output
    assert "requires pip-tools==7.6.1" in output
    assert {name: (requirements_dir / name).read_bytes() for name in GENERIC_LOCK_FILES} == original
    assert not transaction_dir.exists()


def test_make_clean_removes_full_generic_set_and_target_owned_ml_locks(tmp_path: Path) -> None:
    requirements_dir = _create_isolated_lock_lane(tmp_path)
    _write_complete_previous_lock_set(requirements_dir)
    ml_names = (
        "ml-core-darwin-x86_64.txt",
        "ml-core-darwin-arm64.txt",
        "ml-core-linux.txt",
        "ml-sam2.txt",
    )
    for name in ml_names:
        (requirements_dir / name).write_text("example==1.0\n", encoding="utf-8")

    result = subprocess.run(
        ["make", "clean"],
        cwd=requirements_dir,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert all(not (requirements_dir / name).exists() for name in GENERIC_LOCK_FILES)
    assert all(not (requirements_dir / name).exists() for name in ml_names)


def test_compile_ml_layers_refuses_broad_target_owned_regeneration() -> None:
    body = _target_body("compile-ml-layers")

    assert "target-owned ML locks require explicit authoritative-lane commands" in body
    assert "compile-ml-darwin-arm64" in body
    assert "compile-ml-linux-x86_64" not in body


def _write_fake_uname(fakebin: Path, *, system: str, machine: str) -> None:
    (fakebin / "uname").write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  -s) echo {system} ;;\n"
        f"  -m) echo {machine} ;;\n"
        "  *) echo unsupported >&2; exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    (fakebin / "uname").chmod(0o755)


def test_darwin_arm64_target_fails_closed_off_lane(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    requirements_dir = repo_root / "requirements"
    requirements_dir.mkdir(parents=True)
    (requirements_dir / "Makefile").write_text(MAKEFILE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    _write_fake_uname(fakebin, system="Linux", machine="x86_64")

    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(
        ["make", "compile-ml-darwin-arm64"],
        cwd=requirements_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "authoritative only on native Darwin arm64" in (result.stdout + result.stderr)


def test_retired_linux_x86_64_target_exits_nonzero_with_retired_message(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    requirements_dir = repo_root / "requirements"
    requirements_dir.mkdir(parents=True)
    (requirements_dir / "Makefile").write_text(MAKEFILE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        ["make", "compile-ml-linux-x86_64"],
        cwd=requirements_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Linux ML lockfiles are retired unsupported manifests" in (result.stdout + result.stderr)


def test_retired_darwin_x86_64_target_exits_nonzero_with_retired_message(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    requirements_dir = repo_root / "requirements"
    requirements_dir.mkdir(parents=True)
    (requirements_dir / "Makefile").write_text(MAKEFILE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        ["make", "compile-ml-darwin-x86_64"],
        cwd=requirements_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "macOS Intel ML lockfiles are retired unsupported manifests" in (result.stdout + result.stderr)
