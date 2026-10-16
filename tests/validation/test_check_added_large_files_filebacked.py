"""Transport and unchanged policy oracles in private synthetic Git fixtures."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def adapter(monkeypatch):
    """Core CI need not install the remote hook; its helper API is explicit."""
    package_name = "pre_commit_hooks"
    utility_name = package_name + ".util"
    package = ModuleType(package_name)
    setattr(package, "__path__", [])
    utility = ModuleType(utility_name)

    def added_files():
        # Same upstream command/split semantics; private fixture cwd only.
        value = subprocess.check_output(
            ("git", "diff", "--staged", "--name-only", "--diff-filter=A"),
            stderr=subprocess.PIPE,
            timeout=10,
        ).decode()
        return set(value.splitlines())

    def zsplit(value):
        value = value.strip("\0")
        return value.split("\0") if value else []

    setattr(utility, "added_files", added_files)
    setattr(utility, "zsplit", zsplit)
    with monkeypatch.context() as scoped:
        scoped.setitem(sys.modules, package_name, package)
        scoped.setitem(sys.modules, utility_name, utility)
        name = "tested_filebacked_large_files_adapter"
        path = Path(__file__).resolve().parents[2] / "scripts/validation/check_added_large_files_filebacked.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        scoped.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        yield module


@pytest.fixture
def scratch_git(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("Git is required for isolated staged/attribute policy fixtures")
    repo = tmp_path / "repo"
    repo.mkdir()
    templates = tmp_path / "empty-templates"
    templates.mkdir()
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_CONFIG",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_ATTR_SOURCE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.attributesFile")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", os.devnull)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    monkeypatch.chdir(repo)
    subprocess.run(
        ["git", "init", "--template=" + str(templates)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10
    )

    def stage(name, size):
        with open(name, "wb") as f:
            f.truncate(size)
        subprocess.run(["git", "add", "--", name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        return name

    return repo, stage


@pytest.mark.parametrize("size,expected", [(5_000 * 1024, 0), (5_000 * 1024 + 1, 1)])
def test_real_staged_max5000_boundary(adapter, scratch_git, capsys, size, expected):
    _repo, stage = scratch_git
    name = stage("boundary.bin", size)
    assert adapter.main(["--maxkb=5000", name]) == expected
    output = capsys.readouterr().out
    assert output == ("boundary.bin (5001 KB) exceeds 5000 KB.\n" if expected else "")


def test_unstaged_large_file_is_excluded_by_original_default(adapter, scratch_git):
    _repo, stage = scratch_git
    staged = stage("small.bin", 1)
    with open("unstaged.bin", "wb") as f:
        f.truncate(5_000 * 1024 + 1)
    assert adapter.main(["--maxkb=5000", staged, "unstaged.bin"]) == 0
    assert adapter.main(["--maxkb=5000", "--enforce-all", "unstaged.bin"]) == 1


def test_real_lfs_attribute_excludes_large_staged_path(adapter, scratch_git):
    repo, stage = scratch_git
    name = stage("lfs-fixture.bin", 5_000 * 1024 + 1)
    # Add first, then apply a working-tree attribute: no real LFS clean filter,
    # network dependency or LFS-object acceptance is implied.
    (repo / ".gitattributes").write_text("lfs-fixture.bin filter=lfs\n", encoding="utf-8")
    assert adapter.main(["--maxkb=5000", name]) == 0
    assert adapter.main(["--maxkb=5000", "--enforce-all", name]) == 0


@pytest.mark.parametrize(
    "name", ["space name.bin", "line\nname.bin", "carriage\rname.bin", "line\r\nname.bin", "café-撮影.bin"]
)
def test_actual_attribute_transport_roundtrips_special_names(adapter, scratch_git, name):
    _repo, stage = scratch_git
    stage(name, 5_000 * 1024 + 1)
    selected = {name}
    adapter.filter_lfs_files(selected)
    assert selected == {name}
    # Exercise the retained explicit option; upstream added_files() remains
    # line/quote based, so this is NOT a default staged-selector guarantee for
    # newline or quoted Unicode names.
    assert adapter.main(["--maxkb=5000", "--enforce-all", name]) == 1


def test_only_attribute_stdout_changes_and_temp_closes(adapter, monkeypatch):
    handles, calls = [], []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert argv == ("git", "check-attr", "filter", "-z", "--stdin")
        assert kwargs["stdout"] is not subprocess.PIPE
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert kwargs["encoding"] == "utf-8" and kwargs["check"] is True
        assert set(kwargs["input"].split("\0")) == {"plain", "lfs"}
        assert "stdin" not in kwargs
        handle = kwargs["stdout"]
        handles.append(handle)
        os.write(handle.fileno(), b"plain\0filter\0unspecified\0lfs\0filter\0lfs\0")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adapter.subprocess, "run", run)
    selected = {"plain", "lfs"}
    adapter.filter_lfs_files(selected)
    assert selected == {"plain"} and len(calls) == 1
    assert all(handle.closed for handle in handles)


def test_git_attribute_failure_propagates_and_temp_closes(adapter, monkeypatch):
    handles = []
    error = subprocess.CalledProcessError(128, ("git", "check-attr"))

    def failure(_argv, **kwargs):
        handles.append(kwargs["stdout"])
        raise error

    monkeypatch.setattr(adapter.subprocess, "run", failure)
    selected = {"a"}
    with pytest.raises(subprocess.CalledProcessError) as caught:
        adapter.filter_lfs_files(selected)
    assert caught.value is error and selected == {"a"}
    assert all(handle.closed for handle in handles)


@pytest.mark.parametrize(
    "response",
    [
        b"",
        b"a\0filter\0",
        b"a\0filter\0lfs\0",
        b"a\0filter\0lfs\0b\0filter\0unspecified",
        b"a\0filter\0lfs\0a\0filter\0unspecified\0",
        b"a\0filter\0lfs\0outside\0filter\0unspecified\0",
        b"a\0different\0lfs\0b\0filter\0unspecified\0",
        b"a\0filter\0lfs\0b\0filter\0unspecified\0extra\0filter\0lfs\0",
    ],
)
def test_incomplete_duplicate_unknown_or_wrong_key_table_refuses_before_removal(adapter, monkeypatch, response):
    handles = []

    def run(_argv, **kwargs):
        handles.append(kwargs["stdout"])
        os.write(kwargs["stdout"].fileno(), response)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adapter.subprocess, "run", run)
    selected = {"a", "b"}
    with pytest.raises(ValueError, match="Git filter attribute"):
        adapter.filter_lfs_files(selected)
    assert selected == {"a", "b"}
    assert all(handle.closed for handle in handles)


def test_invalid_utf8_response_fails_closed(adapter, monkeypatch):
    def run(_argv, **kwargs):
        os.write(kwargs["stdout"].fileno(), b"a\0filter\0\xff\0")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adapter.subprocess, "run", run)
    with pytest.raises(UnicodeDecodeError):
        adapter.filter_lfs_files({"a"})


@pytest.mark.parametrize("value", ["", "unspecified", "unset", "lfs-like"])
def test_only_exact_lfs_value_is_removed(adapter, monkeypatch, value):
    def run(_argv, **kwargs):
        os.write(kwargs["stdout"].fileno(), ("a\0filter\0" + value + "\0").encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adapter.subprocess, "run", run)
    selected = {"a"}
    adapter.filter_lfs_files(selected)
    assert selected == {"a"}


def test_empty_filter_does_not_invoke_git(adapter, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Empty attribute selection must not spawn Git")

    monkeypatch.setattr(adapter.subprocess, "run", unexpected)
    adapter.filter_lfs_files(set())


def test_missing_added_file_stat_error_is_not_ignored(adapter, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def run(_argv, **kwargs):
        os.write(kwargs["stdout"].fileno(), b"missing\0filter\0unspecified\0")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adapter.subprocess, "run", run)
    monkeypatch.setattr(adapter, "added_files", lambda: {"missing"})
    with pytest.raises(FileNotFoundError):
        adapter.main(["--maxkb=5000", "missing"])


def test_duplicate_cli_paths_query_and_check_once(adapter, scratch_git):
    _repo, stage = scratch_git
    name = stage("same.bin", 1)
    assert adapter.main(["--maxkb=5000", name, name]) == 0
