"""Fail-closed contracts for the additive editorial runtime installer."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import platform
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("editorial_runtime_contract_tests", ROOT / "scripts/setup/editorial_runtime.py")
assert SPEC and SPEC.loader
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def locked_pip() -> bytes:
    return f"pip==26.2.1 --hash=sha256:{'a' * 64}\n".encode()


@pytest.mark.parametrize(
    "system,machine,version", [("Linux", "arm64", (3, 12)), ("Darwin", "x86_64", (3, 12)), ("Darwin", "arm64", (3, 11))]
)
def test_unsupported_target_fails_before_install(tmp_path, monkeypatch, system, machine, version):
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)
    monkeypatch.setattr(runtime.sys, "version_info", version)
    target = tmp_path / "never-created"
    with pytest.raises(ValueError, match="Darwin arm64.*3.12"):
        runtime.install(target)
    assert not target.exists()


def test_old_macos_fails_before_download(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(runtime.sys, "version_info", (3, 12))
    monkeypatch.setattr(runtime.platform, "mac_ver", lambda: ("13.7", (), ""))
    with pytest.raises(ValueError, match="macOS 14"):
        runtime.require_target()


@pytest.mark.parametrize(
    "extra",
    [
        "-r other.txt",
        "--index-url https://evil.invalid/simple",
        "pkg @ https://evil.invalid/x.whl",
        "pkg==1",
        f"pkg==1 ; python_version > '3' --hash=sha256:{'b'*64}",
        locked_pip().decode(),
    ],
)
def test_lock_rejects_uncontrolled_requirements(extra):
    with pytest.raises(ValueError):
        runtime.parse_lock(locked_pip() + extra.encode())


def test_checked_in_contract_covers_exact_closure():
    contract, lock, packages = runtime.load_contract()
    assert set(contract["artifacts"]) == set(packages)
    assert {"rawpy", "reportlab", "exifread", "piexif", "imagecodecs", "pip"} <= set(packages)
    assert runtime.digest(lock) == contract["lock_sha256"]


@pytest.fixture(name="copied_contract")
def copied_contract_fixture(tmp_path, monkeypatch):
    for variable in ("INPUT_PATH", "LOCK_PATH", "CONTRACT_PATH", "GENERIC_PATH"):
        original = getattr(runtime, variable)
        copy = tmp_path / variable
        copy.write_bytes(original.read_bytes())
        monkeypatch.setattr(runtime, variable, copy)
    return runtime


@pytest.mark.parametrize("variable", ["INPUT_PATH", "LOCK_PATH"])
def test_changed_source_or_lock_fails_before_runtime(copied_contract, variable):
    path = getattr(copied_contract, variable)
    path.write_bytes(path.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="input/lock drift"):
        copied_contract.load_contract()


def test_changed_core_pin_requires_regeneration(copied_contract):
    path = copied_contract.GENERIC_PATH
    path.write_text(path.read_text().replace("numpy==2.4.6", "numpy==2.4.5"))
    with pytest.raises(ValueError, match="core constraints drift"):
        copied_contract.load_contract()


def test_wrong_artifact_hash_or_missing_member_fails(copied_contract):
    contract = json.loads(copied_contract.CONTRACT_PATH.read_bytes())
    contract["artifacts"]["pip"]["sha256"] = "0" * 64
    copied_contract.CONTRACT_PATH.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="hash conflicts"):
        copied_contract.load_contract()
    del contract["artifacts"]["pip"]
    copied_contract.CONTRACT_PATH.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="complete closure"):
        copied_contract.load_contract()


@pytest.mark.parametrize(
    "url",
    [
        "http://files.pythonhosted.org/packages/a.whl",
        "https://evil.invalid/packages/a.whl",
        "https://files.pythonhosted.org:443/packages/a.whl",
        "https://user@files.pythonhosted.org/packages/a.whl",
        "https://files.pythonhosted.org/packages/a.whl?token=x",
        "https://files.pythonhosted.org/packages/a.whl#x",
        "https://files.pythonhosted.org/packages/a.tar.gz",
        "https://files.pythonhosted.org/packages/%2fescape.whl",
    ],
)
def test_untrusted_or_nonwheel_urls_are_rejected(url):
    with pytest.raises(ValueError):
        runtime.wheel_url(url)


def test_redirect_is_rejected():
    with pytest.raises(ValueError, match="redirect"):
        runtime.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://evil.invalid/a.whl")


@pytest.mark.parametrize(
    "expected,limit,message", [("0" * 64, 100, "hash mismatch"), (runtime.digest(b"wheel"), 4, "artifact bound")]
)
def test_download_hash_and_size_enforced(tmp_path, monkeypatch, expected, limit, message):
    class Opener:
        def open(self, _url, timeout):
            assert timeout == 30
            return io.BytesIO(b"wheel")

    monkeypatch.setattr(runtime.urllib.request, "build_opener", lambda *_args: Opener())
    monkeypatch.setattr(runtime, "MAX_WHEEL_BYTES", limit)
    with pytest.raises(ValueError, match=message):
        runtime.download_wheel("https://files.pythonhosted.org/packages/package.whl", expected, tmp_path)


def test_unsafe_runtime_root_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        runtime.safe_root(link / "nested", create=True)
    real.chmod(0o777)
    with pytest.raises(ValueError, match="write access"):
        runtime.safe_root(real)


@pytest.mark.parametrize(
    "target", ["../escape", "/private/tmp/escape", "generations/../../escape", "generations/not-a-generation"]
)
def test_current_link_must_be_contained_generation(tmp_path, target):
    (tmp_path / "current").symlink_to(target)
    with pytest.raises(ValueError, match="contained generation"):
        runtime.installed_generation(tmp_path)


def test_failed_install_preserves_current_generation(tmp_path, monkeypatch):
    old = tmp_path / "generations" / ("a" * 32)
    old.mkdir(parents=True)
    current = tmp_path / "current"
    current.symlink_to(Path("generations") / old.name)
    monkeypatch.setattr(runtime, "require_target", lambda: None)
    monkeypatch.setattr(
        runtime, "load_contract", lambda: ({"artifacts": {"pip": {"url": "url", "sha256": "hash"}}}, locked_pip(), {})
    )

    def fail(*_args):
        raise ValueError("simulated network failure")

    monkeypatch.setattr(runtime, "download_wheel", fail)
    with pytest.raises(ValueError, match="simulated network"):
        runtime.install(tmp_path)
    assert current.resolve() == old
    assert old.exists()
    assert len(list((tmp_path / "generations").iterdir())) == 2


@pytest.fixture(name="simulated_venv")
def simulated_venv_fixture(tmp_path):
    site = tmp_path / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    builder = Path(sys._base_executable).resolve()
    (tmp_path / "bin/python").symlink_to(builder)
    (tmp_path / "pyvenv.cfg").write_text(
        f"home = {builder.parent}\ninclude-system-site-packages = false\n"
        f"version = {platform.python_version()}\nexecutable = {builder}\n"
    )
    return tmp_path


def test_interpreter_and_startup_hook_drift_rejected(simulated_venv):
    runtime.audit_interpreter(simulated_venv)
    injection = simulated_venv / "lib/python3.12/site-packages/injected.pth"
    injection.write_text("import attacker\n")
    with pytest.raises(ValueError, match="startup hooks"):
        runtime.audit_interpreter(simulated_venv)
    injection.unlink()
    (simulated_venv / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    with pytest.raises(ValueError, match="venv configuration"):
        runtime.audit_interpreter(simulated_venv)


def test_wrong_interpreter_rejected(simulated_venv):
    interpreter = simulated_venv / "bin/python"
    interpreter.unlink()
    interpreter.symlink_to("/bin/sh")
    with pytest.raises(ValueError, match="trusted Python builder"):
        runtime.audit_interpreter(simulated_venv)


def test_installed_lock_drift_rejected_before_execution(tmp_path, monkeypatch):
    (tmp_path / "runtime.json").write_text("{}")
    (tmp_path / "requirements.txt").write_bytes(b"changed")
    monkeypatch.setattr(runtime, "run_checked", lambda *_args, **_kwargs: pytest.fail("must not execute altered runtime"))
    with pytest.raises(ValueError, match="runtime lock/contract drift"):
        runtime.verify_generation(tmp_path, {}, locked_pip())


def test_python_and_pip_environment_overrides_are_removed(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/malicious")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://evil.invalid")
    result = runtime.clean_environment()
    assert "PYTHONPATH" not in result and "PIP_EXTRA_INDEX_URL" not in result
    assert result["PIP_CONFIG_FILE"] == os.devnull


@pytest.fixture(name="authenticated_payload")
def authenticated_payload_fixture(tmp_path):
    import zipfile

    site = tmp_path / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    module = site / "pip/__main__.py"
    module.parent.mkdir()
    module.write_bytes(b"print('trusted wheel source')\n")
    wheel = wheels / "pip-26.2.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pip/__main__.py", module.read_bytes())
    contract = {
        "artifacts": {
            "pip": {"url": "https://files.pythonhosted.org/packages/" + wheel.name, "sha256": runtime.file_digest(wheel)}
        }
    }
    return tmp_path, module, wheel, contract


def test_installed_package_tamper_rejected_before_pip_execution(authenticated_payload, monkeypatch):
    generation, module, _wheel, contract = authenticated_payload
    (generation / "runtime.json").write_text(json.dumps(contract))
    (generation / "requirements.txt").write_bytes(locked_pip())
    monkeypatch.setattr(runtime, "audit_interpreter", lambda _path: None)
    monkeypatch.setattr(runtime, "invoke_runtime", lambda *_args, **_kwargs: pytest.fail("tampered pip must not execute"))
    module.write_text("raise RuntimeError('untrusted pip code')\n")
    with pytest.raises(ValueError, match="installed wheel payload mismatch"):
        runtime.verify_generation(generation, contract, locked_pip())


def test_retained_wheel_tamper_rejected(authenticated_payload):
    generation, _module, wheel, contract = authenticated_payload
    wheel.write_bytes(b"modified wheel")
    with pytest.raises(ValueError, match="retained wheel hash"):
        runtime.audit_payload(generation, contract)


def test_extra_package_shadow_or_bytecode_only_payload_rejected(authenticated_payload):
    generation, module, _wheel, contract = authenticated_payload
    runtime.audit_payload(generation, contract)
    extra = module.parent / "unreviewed.pyc"
    extra.write_bytes(b"bytecode-only injection")
    with pytest.raises(ValueError, match="Unexpected editorial package payload"):
        runtime.audit_payload(generation, contract)


def test_old_bytecode_cache_is_not_on_invocation_search_path(authenticated_payload, monkeypatch):
    generation, module, _wheel, contract = authenticated_payload
    cache = module.parent / "__pycache__"
    cache.mkdir()
    (cache / "__main__.cpython-312.pyc").write_bytes(b"stale bytecode")
    runtime.audit_payload(generation, contract)
    prefixes = []

    def capture(command, **kwargs):
        assert command[:4] == [str(generation / "bin/python"), "-I", "-B", "-X"]
        prefix = Path(command[4].split("=", 1)[1])
        assert prefix.is_dir() and not list(prefix.iterdir())
        assert generation not in prefix.parents
        prefixes.append(prefix)
        assert kwargs["check"] is True

    monkeypatch.setattr(runtime.subprocess, "run", capture)
    runtime.invoke_runtime(generation, ["-m", "pip", "check"])
    runtime.invoke_runtime(generation, ["-m", "pip", "check"])
    assert prefixes[0] != prefixes[1]
    assert not any(prefix.exists() for prefix in prefixes)


def test_wrong_compiler_version_is_rejected(monkeypatch):
    monkeypatch.setattr(runtime, "require_target", lambda: None)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda _name: "0")
    with pytest.raises(ValueError, match="compiler requires pip==26.2.1"):
        runtime.compile_lock()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is a POSIX admission case")
@pytest.mark.parametrize("name", ["runtime.json", "requirements.txt"])
def test_nonregular_evidence_rejected_without_reading(tmp_path, name):
    for path, content in ((tmp_path / "runtime.json", b"{}"), (tmp_path / "requirements.txt", locked_pip())):
        if path.name == name:
            os.mkfifo(path)
        else:
            path.write_bytes(content)
    with pytest.raises(ValueError, match="regular files"):
        runtime.verify_generation(tmp_path, {}, locked_pip())


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is a POSIX admission case")
def test_nonregular_retained_wheel_rejected_without_hashing(authenticated_payload):
    generation, _module, wheel, contract = authenticated_payload
    wheel.unlink()
    os.mkfifo(wheel)
    with pytest.raises(ValueError, match="bounded regular file"):
        runtime.audit_payload(generation, contract)


def test_compiler_uses_frozen_source_and_constraint_bytes(tmp_path, monkeypatch):
    """A transient live-file edit cannot be compiled under the original digest."""
    import subprocess

    for variable, relative in (
        ("INPUT_PATH", "editorial.in"),
        ("GENERIC_PATH", "all.txt"),
        ("LOCK_PATH", "locks/editorial.txt"),
        ("CONTRACT_PATH", "locks/editorial.json"),
    ):
        monkeypatch.setattr(runtime, variable, tmp_path / relative)
    original = b"pip==26.2.1\n"
    runtime.INPUT_PATH.write_bytes(original)
    runtime.GENERIC_PATH.write_bytes(b"# original constraints\n")
    monkeypatch.setattr(runtime, "require_target", lambda: None)
    monkeypatch.setattr(runtime.importlib.metadata, "version", runtime.COMPILER.get)

    def resolve(command, **_kwargs):
        if "piptools" in command:
            runtime.INPUT_PATH.write_bytes(original + b"unexpected==1.0\n")
            runtime.GENERIC_PATH.write_bytes(b"pip==0\n")
            source = Path(command[-1])
            assert source.read_bytes() == original
            assert source.with_name("all.txt").read_bytes() == b"# original constraints\n"
            Path(command[command.index("--output-file") + 1]).write_bytes(source.read_bytes())
            runtime.INPUT_PATH.write_bytes(original)
            runtime.GENERIC_PATH.write_bytes(b"# original constraints\n")
        else:
            assert "--report" in command
            report = {
                "install": [
                    {
                        "metadata": {"name": "pip", "version": "26.2.1"},
                        "download_info": {
                            "url": "https://files.pythonhosted.org/packages/pip-26.2.1-py3-none-any.whl",
                            "archive_info": {"hashes": {"sha256": "a" * 64}},
                        },
                    }
                ]
            }
            Path(command[command.index("--report") + 1]).write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime.subprocess, "run", resolve)
    runtime.compile_lock()
    assert set(runtime.load_contract()[2]) == {"pip"}
    assert runtime.INPUT_PATH.read_bytes() == original


def test_download_transfer_budget_checked_after_each_read(tmp_path, monkeypatch):
    class Opener:
        def open(self, _url, timeout):
            assert timeout == 30
            return io.BytesIO(b"wheel")

    clock = iter([0, 301])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(runtime.urllib.request, "build_opener", lambda *_args: Opener())
    with pytest.raises(ValueError, match="transfer budget"):
        runtime.download_wheel("https://files.pythonhosted.org/packages/package.whl", runtime.digest(b"wheel"), tmp_path)
