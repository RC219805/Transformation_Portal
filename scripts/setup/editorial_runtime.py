#!/usr/bin/env python3
"""Manage the isolated Darwin arm64 / Python 3.12 editorial runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "requirements/editorial.in"
LOCK_PATH = REPO_ROOT / "requirements/locks/editorial-darwin-arm64-py312.txt"
CONTRACT_PATH = REPO_ROOT / "requirements/locks/editorial-darwin-arm64-py312.json"
GENERIC_PATH = REPO_ROOT / "requirements/all.txt"
SMOKE_PATH = REPO_ROOT / "scripts/validation/check_editorial_runtime.py"
TOOL_PATH = REPO_ROOT / "tools/ad_editorial_post_pipeline.py"
COMPILER = {"pip": "26.2.1", "pip-tools": "7.6.1", "click": "8.4.2"}
SCHEMA = "tp.editorial.runtime.v1"
MAX_WHEEL_BYTES = 256 * 1024 * 1024
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)$")
HASH = re.compile(r"^--hash=sha256:([a-f0-9]{64})$")


def digest(data: bytes) -> str:
    """Hash source or artifact bytes for the runtime contract."""
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    """Hash installed files without loading native libraries into memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Editorial payload must be a regular file: {path}")
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def normalize(name: str) -> str:
    """Use the package-index normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def require_target() -> None:
    """Only the compiled native interpreter/host lane is supported."""
    if (platform.system(), platform.machine(), sys.implementation.name, sys.version_info[:2]) != (
        "Darwin",
        "arm64",
        "cpython",
        (3, 12),
    ):
        raise ValueError("Editorial runtime requires native Darwin arm64 and CPython 3.12")
    if int(platform.mac_ver()[0].split(".", 1)[0] or "0") < 14:
        raise ValueError("Editorial native wheels require macOS 14 or newer")


def parse_lock(data: bytes) -> dict[str, tuple[str, set[str]]]:
    """Reject URLs, includes, options, markers, duplicates, and unhashed pins."""
    packages = {}
    lines = [line.split("#", 1)[0].strip() for line in data.decode("utf-8").splitlines()]
    for line in "\n".join(lines).replace("\\\n", " ").splitlines():
        tokens = line.split()
        if not tokens:
            continue
        pin = PIN.fullmatch(tokens[0])
        hashes = [HASH.fullmatch(token) for token in tokens[1:]]
        if not pin or not hashes or not all(hashes):
            raise ValueError(f"Editorial lock requires exact pins and SHA256 hashes: {tokens[0]}")
        name = normalize(pin[1])
        if name in packages:
            raise ValueError(f"Duplicate editorial lock package: {name}")
        packages[name] = (pin[2], {match[1] for match in hashes if match})
    if not packages or packages.get("pip", (None,))[0] != COMPILER["pip"]:
        raise ValueError("Editorial lock must include the governed pip bootstrap")
    return packages


def core_pins(packages: dict) -> dict[str, str]:
    """Compare only closure members with the governed generic constraints."""
    pins = {}
    for line in GENERIC_PATH.read_text(encoding="utf-8").splitlines():
        pin = PIN.fullmatch(line.split(";", 1)[0].strip())
        if pin and normalize(pin[1]) in packages:
            pins[normalize(pin[1])] = pin[2]
    return pins


def load_contract() -> tuple[dict, bytes, dict]:
    """Check input/lock identity and shared core versions before any execution."""
    contract = json.loads(CONTRACT_PATH.read_bytes())
    lock = LOCK_PATH.read_bytes()
    packages = parse_lock(lock)
    if contract.get("schema") != SCHEMA or contract.get("target") != "darwin-arm64-py312":
        raise ValueError("Invalid editorial runtime contract")
    if contract.get("compiler") != COMPILER:
        raise ValueError("Editorial compiler contract drift")
    if contract.get("lock_sha256") != digest(lock) or contract.get("input_sha256") != digest(INPUT_PATH.read_bytes()):
        raise ValueError("Editorial input/lock drift; regenerate the target-owned lock")
    if contract.get("core_pins") != core_pins(packages):
        raise ValueError("Editorial core constraints drift; regenerate the target-owned lock")
    for name, version in contract["core_pins"].items():
        if packages[name][0] != version:
            raise ValueError(f"Editorial lock conflicts with governed core pin: {name}")
    artifacts = contract.get("artifacts", {})
    if set(artifacts) != set(packages):
        raise ValueError("Editorial artifact contract must cover the complete closure")
    for name, artifact in artifacts.items():
        wheel_url(artifact["url"])
        if artifact["sha256"] not in packages[name][1]:
            raise ValueError(f"Editorial artifact hash conflicts with lock: {name}")
    return contract, lock, packages


def clean_environment() -> dict[str, str]:
    """Ignore ambient Python startup and pip index/configuration overrides."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "PIP_"))}
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_CACHE_DIR="1")
    return env


def run_checked(command: list[str], *, timeout: int = 600, capture: bool = False) -> subprocess.CompletedProcess:
    """Run argv directly with bounded duration and the cleaned environment."""
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=clean_environment(),
        check=True,
        timeout=timeout,
        text=True,
        capture_output=capture,
    )


def safe_root(path: Path, *, create: bool = False) -> Path:
    """Reject symlinked runtime ancestors and roots writable by other users."""
    path = Path(os.path.abspath(path.expanduser()))
    for part in [*reversed(path.parents), path]:
        if part.is_symlink():
            raise ValueError(f"Editorial runtime path must not contain symlinks: {part}")
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.stat()
    if not path.is_dir() or metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
        raise ValueError("Editorial runtime root must be an owned directory without group/other write access")
    return path


def wheel_url(url: str) -> str:
    """Accept only ordinary HTTPS PyPI artifact URLs, without credentials."""
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/packages/")
    ):
        raise ValueError("Wheel URL must be a trusted files.pythonhosted.org HTTPS artifact")
    filename = parsed.path.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename):
        raise ValueError("Editorial runtime accepts wheel artifacts only")
    return filename


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Artifact URLs are immutable; reject redirects before fetching another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Editorial wheel downloads may not redirect")


def download_wheel(url: str, expected_hash: str, directory: Path) -> Path:
    """Fetch one bounded, hash-verified wheel from the reviewed resolver report."""
    filename = wheel_url(url)
    target = directory / filename
    hasher = hashlib.sha256()
    total = 0
    started = time.monotonic()
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(url, timeout=30) as response, target.open("xb") as output:
        # read1 returns after one underlying read, unlike read(size), which may
        # keep filling its buffer through many individually successful reads.
        while chunk := response.read1(1024 * 1024):
            if time.monotonic() - started > 300:
                raise ValueError("Editorial wheel download exceeded the five-minute transfer budget")
            total += len(chunk)
            if total > MAX_WHEEL_BYTES:
                raise ValueError("Editorial wheel exceeds the 256 MiB artifact bound")
            hasher.update(chunk)
            output.write(chunk)
    if not total or hasher.hexdigest() != expected_hash:
        raise ValueError(f"Editorial wheel hash mismatch: {filename}")
    return target


def compile_lock() -> None:
    """Compile the closure, then bind hashes to wheels for this native target."""
    require_target()
    for name, version in COMPILER.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Editorial lock compiler requires {name}=={version}")
    from packaging.requirements import Requirement

    input_bytes = INPUT_PATH.read_bytes()
    generic_bytes = GENERIC_PATH.read_bytes()
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tp-editorial-lock-") as temporary:
        staged = Path(temporary) / "inputs"
        staged.mkdir()
        staged_input = staged / "editorial.in"
        staged_input.write_bytes(input_bytes)
        (staged / "all.txt").write_bytes(generic_bytes)
        output = Path(temporary) / "resolved.txt"
        environment = clean_environment()
        environment["CUSTOM_COMPILE_COMMAND"] = "python scripts/setup/editorial_runtime.py lock"
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "piptools",
                "compile",
                "--no-config",
                "--cache-dir",
                temporary,
                "--allow-unsafe",
                "--strip-extras",
                "--no-emit-options",
                "--no-emit-index-url",
                "--no-emit-trusted-host",
                "--pip-args=--only-binary=:all: --index-url=https://pypi.org/simple",
                "--output-file",
                str(output),
                str(staged_input),
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            timeout=900,
        )
        resolved = {}
        for line in output.read_text(encoding="utf-8").splitlines():
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            match = PIN.fullmatch(text.split(";", 1)[0].strip())
            if not match or requirement.extras or normalize(match[1]) in resolved:
                raise ValueError("Compiler emitted an unsupported or duplicate requirement")
            resolved[normalize(match[1])] = match[2]
        output.write_text("".join(f"{name}=={version}\n" for name, version in sorted(resolved.items())), encoding="utf-8")
        report_path = Path(temporary) / "resolution.json"
        run_checked(
            [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--no-deps",
                "--only-binary=:all:",
                "--index-url",
                "https://pypi.org/simple",
                "--report",
                str(report_path),
                "-r",
                str(output),
            ]
        )
        artifacts = {}
        for item in json.loads(report_path.read_bytes())["install"]:
            name = normalize(item["metadata"]["name"])
            info = item["download_info"]
            sha256 = info["archive_info"]["hashes"]["sha256"]
            wheel_url(info["url"])
            if (
                name in artifacts
                or item["metadata"]["version"] != resolved.get(name)
                or not re.fullmatch(r"[a-f0-9]{64}", sha256)
            ):
                raise ValueError("Target wheel report conflicts with the compiled closure")
            artifacts[name] = {"url": info["url"], "sha256": sha256}
        if set(artifacts) != set(resolved):
            raise ValueError("Target wheel report is incomplete")
        header = (
            "# Generated by scripts/setup/editorial_runtime.py lock; do not hand-edit.\n"
            "# Target: native Darwin arm64 / CPython 3.12.\n"
            "# Compiler: pip 26.2.1 / pip-tools 7.6.1 / Click 8.4.2.\n"
            "# Hashes bind the target-native wheels recorded in the companion JSON.\n\n"
        )
        lock = (
            header
            + "".join(f"{name}=={resolved[name]} --hash=sha256:{artifacts[name]['sha256']}\n" for name in sorted(resolved))
        ).encode("utf-8")
        packages = parse_lock(lock)
        pins = core_pins(packages)
        if any(packages[name][0] != version for name, version in pins.items()):
            raise ValueError("Compiled editorial core versions differ from governed constraints")
        if INPUT_PATH.read_bytes() != input_bytes or GENERIC_PATH.read_bytes() != generic_bytes:
            raise ValueError("Editorial input/constraints changed during compilation")
        contract = {
            "schema": SCHEMA,
            "target": "darwin-arm64-py312",
            "compiler": COMPILER,
            "input_sha256": digest(input_bytes),
            "lock_sha256": digest(lock),
            "core_pins": pins,
            "artifacts": artifacts,
        }
        # Publishing either half alone fails closed; compilation never edits generic/ML locks.
        LOCK_PATH.write_bytes(lock)
        CONTRACT_PATH.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_contract()


def installed_generation(root: Path) -> Path:
    """Admit only the installer-owned current generation link inside this root."""
    root = safe_root(root)
    current = root / "current"
    if not current.is_symlink():
        raise ValueError("Editorial current must be an installer-owned generation symlink")
    relative = Path(os.readlink(current))
    if len(relative.parts) != 2 or relative.parts[0] != "generations" or not re.fullmatch(r"[a-f0-9]{32}", relative.name):
        raise ValueError("Editorial current link must target a contained generation")
    return safe_root(root / relative)


def audit_interpreter(generation: Path) -> None:
    """Admit the current trusted builder and reject injected Python startup hooks."""
    safe_root(generation)
    builder = Path(sys._base_executable).resolve()
    python = generation / "bin/python"
    if not python.is_symlink() or python.resolve() != builder:
        raise ValueError("Editorial interpreter differs from the current trusted Python builder; reinstall")
    config = generation / "pyvenv.cfg"
    if config.is_symlink() or not config.is_file():
        raise ValueError("Editorial pyvenv.cfg must be a regular file")
    values = dict(line.split(" = ", 1) for line in config.read_text(encoding="utf-8").splitlines() if " = " in line)
    if (
        values.get("include-system-site-packages") != "false"
        or values.get("version") != platform.python_version()
        or Path(values.get("executable", "")).resolve() != builder
        or Path(values.get("home", "")).resolve() != builder.parent
    ):
        raise ValueError("Editorial venv configuration differs from the isolated trusted builder")
    site = generation / "lib/python3.12/site-packages"
    for part in (generation / "bin", generation / "lib", site.parent, site):
        safe_root(part)
    for parent, directories, files in os.walk(site, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError("Editorial site-packages may not contain symlinks")
            if not path.is_dir() and not path.is_file():
                raise ValueError("Editorial site-packages may contain only regular files and directories")
            if path.parent == site and (name.endswith(".pth") or name.startswith(("sitecustomize", "usercustomize"))):
                raise ValueError("Editorial runtime may not contain Python startup hooks")


def audit_payload(generation: Path, contract: dict) -> None:
    """Authenticate installed package bytes against retained, hash-bound wheels."""
    site = generation / "lib/python3.12/site-packages"
    wheelhouse = safe_root(generation / "wheels")
    expected = set()
    metadata = set()
    for artifact in contract["artifacts"].values():
        wheel = wheelhouse / wheel_url(artifact["url"])
        if wheel.is_symlink() or not wheel.is_file() or wheel.stat().st_size > MAX_WHEEL_BYTES:
            raise ValueError("Editorial retained wheel must be a bounded regular file")
        if file_digest(wheel) != artifact["sha256"]:
            raise ValueError("Editorial retained wheel hash mismatch")
        with zipfile.ZipFile(wheel) as archive:
            members = archive.infolist()
            if sum(member.file_size for member in members) > 2 * 1024**3:
                raise ValueError("Editorial unpacked wheel exceeds the 2 GiB bound")
            for member in members:
                relative = Path(member.filename)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or any(part.endswith(".data") for part in relative.parts)
                ):
                    raise ValueError("Editorial wheel contains an unsupported relocated or escaping payload")
                if member.is_dir():
                    continue
                if relative in expected:
                    raise ValueError("Editorial wheel payloads overlap")
                expected.add(relative)
                if len(relative.parts) == 2 and relative.parts[0].endswith(".dist-info") and relative.name == "RECORD":
                    # pip rewrites RECORD with installation-specific console scripts and hashes.
                    if not (site / relative).is_file():
                        raise ValueError("Editorial installed RECORD must be a regular file")
                    metadata.add(relative.parent)
                    continue
                with archive.open(member) as source:
                    expected_digest = hashlib.file_digest(source, "sha256").hexdigest()
                target = site / relative
                if not target.is_file() or file_digest(target) != expected_digest:
                    raise ValueError(f"Editorial installed wheel payload mismatch: {relative}")
    for directory in metadata:
        for name, content in (("INSTALLER", b"pip\n"), ("REQUESTED", b"")):
            relative = directory / name
            target = site / relative
            if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
                raise ValueError(f"Editorial installation metadata drift: {relative}")
            expected.add(relative)
    for path in site.rglob("*"):
        if path.is_file() and path.relative_to(site) not in expected:
            # Isolated launches use a new empty pycache prefix, never these caches.
            if path.parent.name == "__pycache__" and path.suffix == ".pyc":
                continue
            raise ValueError(f"Unexpected editorial package payload: {path.relative_to(site)}")


def invoke_runtime(generation: Path, arguments: list[str], *, timeout: int | None = 600, check: bool = True):
    """Ignore existing bytecode caches while running authenticated source/native files."""
    with tempfile.TemporaryDirectory(prefix="tp-editorial-bytecode-") as cache:
        return subprocess.run(
            [str(generation / "bin/python"), "-I", "-B", "-X", f"pycache_prefix={cache}", *arguments],
            cwd=REPO_ROOT,
            env=clean_environment(),
            check=check,
            timeout=timeout,
        )


def verify_generation(generation: Path, contract: dict, lock: bytes, *, raw_file: Path | None = None) -> None:
    """Verify dependency identity, consistency, and actual encoders before use."""
    marker = generation / "runtime.json"
    evidence = (marker, generation / "requirements.txt")
    if any(path.is_symlink() or not path.is_file() for path in evidence):
        raise ValueError("Editorial runtime evidence must be regular files")
    if json.loads(marker.read_bytes()) != contract or (generation / "requirements.txt").read_bytes() != lock:
        raise ValueError("Installed editorial runtime lock/contract drift")
    audit_interpreter(generation)
    audit_payload(generation, contract)
    invoke_runtime(generation, ["-m", "pip", "check"], timeout=120)
    command = [
        str(SMOKE_PATH),
        "--lock",
        str(generation / "requirements.txt"),
        "--expected-prefix",
        str(generation),
    ]
    if raw_file:
        command.extend(["--raw-file", str(raw_file.absolute())])
    invoke_runtime(generation, command, timeout=180)


def install(root: Path, *, raw_file: Path | None = None) -> Path:
    """Build a new generation and publish its pointer only after successful checks."""
    require_target()
    import fcntl

    contract, lock, _ = load_contract()
    root = safe_root(root, create=True)
    descriptor = os.open(root / ".install.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as install_lock:
        fcntl.flock(install_lock, fcntl.LOCK_EX)
        generations = root / "generations"
        safe_root(generations, create=True)
        generation = generations / uuid.uuid4().hex
        generation.mkdir(mode=0o700)
        try:
            wheelhouse = generation / "wheels"
            wheelhouse.mkdir(mode=0o700)
            snapshot = generation / "requirements.txt"
            snapshot.write_bytes(lock)
            wheels = [(artifact["url"], artifact["sha256"]) for artifact in contract["artifacts"].values()]
            downloaded = [download_wheel(url, sha256, wheelhouse) for url, sha256 in wheels]
            run_checked([sys.executable, "-I", "-m", "venv", "--without-pip", str(generation)])
            audit_interpreter(generation)
            pip_wheel = next(path for path in downloaded if path.name.startswith("pip-"))
            # Seed only the hash-verified pip wheel, without ambient site packages or network.
            bootstrap = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('pip',run_name='__main__')"
            invoke_runtime(
                generation,
                [
                    "-c",
                    bootstrap,
                    str(pip_wheel),
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--no-compile",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "--find-links",
                    str(wheelhouse),
                    "-r",
                    str(snapshot),
                ],
            )
            (generation / "runtime.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            verify_generation(generation, contract, lock, raw_file=raw_file)
            if load_contract()[:2] != (contract, lock):
                raise ValueError("Editorial source contract changed during installation")
            current = root / "current"
            if current.exists() and not current.is_symlink():
                raise ValueError("Refusing to replace an existing non-symlink editorial current path")
            temporary_link = root / f".current-{uuid.uuid4().hex}"
            temporary_link.symlink_to(Path("generations") / generation.name)
            temporary_link.replace(current)
        except Exception:
            print(f"Failed editorial generation retained for inspection: {generation}", file=sys.stderr)
            raise
    print(f"Editorial runtime ready: {root / 'current/bin/python'}")
    return generation


def main(argv: list[str] | None = None) -> int:
    """Run bounded install, check, lock-generation, or editorial execution commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=REPO_ROOT / ".runtime/editorial")
    parser.add_argument("--raw-file", type=Path, help="Optional owned RAW fixture for decode smoke")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("lock", help="Regenerate the target lock with the pinned compiler")
    commands.add_parser("install", help="Install and verify a fresh generation; preserve existing generations")
    check = commands.add_parser("check", help="Verify current runtime; no downloads or installations")
    check.add_argument("--contract-only", action="store_true", help="Check source identity and core pins on any host")
    execution = commands.add_parser("run", help="Verify runtime, then run the existing editorial command")
    execution.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "lock":
            compile_lock()
        elif args.command == "install":
            install(args.runtime_root, raw_file=args.raw_file)
        else:
            contract, lock, _ = load_contract()
            if args.command == "check" and args.contract_only:
                print("Editorial source contract: OK")
                return 0
            require_target()
            generation = installed_generation(args.runtime_root)
            verify_generation(generation, contract, lock, raw_file=args.raw_file)
            if args.command == "run":
                arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
                result = invoke_runtime(generation, [str(TOOL_PATH), "run", *arguments], timeout=None, check=False)
                return result.returncode
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"Editorial runtime error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
