"""PluginLoader manifest discovery tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.plugins.loader_test_helpers import (
    isolated_loader,
    write_plugin_json,
    write_plugin_module,
    write_pyproject_manifest,
)
from transformation_portal.plugins.signing import PLUGIN_SIGNATURE_ALGORITHM, sign_manifest

pytestmark = [pytest.mark.unit]


def _write_trust_store(tmp_path: Path, *, key_id: str = "local-dev", secret: str = "test-secret") -> Path:
    trust_store_path = tmp_path / "plugin-trust.json"
    trust_store_path.write_text(json.dumps({"keys": {key_id: secret}}, indent=2) + "\n", encoding="utf-8")
    return trust_store_path


def _sign_plugin_json(manifest_path: Path, *, key_id: str = "local-dev", secret: str = "test-secret") -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["signature_algorithm"] = PLUGIN_SIGNATURE_ALGORITHM
    payload["signature_key_id"] = key_id
    payload["signature"] = sign_manifest(payload, secret=secret)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_discovers_plugin_json_manifest(tmp_path: Path):
    package_dir = tmp_path / "json_package"
    write_plugin_module(
        package_dir,
        "json_plugin",
        class_name="JsonPlugin",
        plugin_name="json_plugin",
        execute_result="json-ok",
    )
    write_plugin_json(
        package_dir,
        name="json_plugin",
        module_name="json_plugin",
        class_name="JsonPlugin",
    )

    discovered = isolated_loader(tmp_path).discover_all()

    assert [plugin.manifest.name for plugin in discovered if plugin.manifest] == ["json_plugin"]
    assert discovered[0].plugin.execute() == "json-ok"
    assert discovered[0].is_valid is True


def test_isolated_loader_discards_relative_env_default_path(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGINS", "env_plugins")

    env_package_dir = tmp_path / "env_plugins" / "env_package"
    write_plugin_module(env_package_dir, "env_plugin", plugin_name="env_plugin")
    write_plugin_json(env_package_dir, name="env_plugin", module_name="env_plugin")

    target_root = tmp_path / "target_plugins"
    target_package_dir = target_root / "target_package"
    write_plugin_module(target_package_dir, "target_plugin", plugin_name="target_plugin")
    write_plugin_json(target_package_dir, name="target_plugin", module_name="target_plugin")

    discovered = isolated_loader(target_root).discover_all()

    assert [plugin.manifest.name for plugin in discovered if plugin.manifest] == ["target_plugin"]


def test_discovers_pyproject_manifest_when_plugin_json_absent(tmp_path: Path):
    package_dir = tmp_path / "pyproject_package"
    write_plugin_module(
        package_dir,
        "pyproject_plugin",
        class_name="PyprojectPlugin",
        plugin_name="pyproject_plugin",
    )
    write_pyproject_manifest(
        package_dir,
        name="pyproject_plugin",
        module_name="pyproject_plugin",
        class_name="PyprojectPlugin",
    )

    discovered = isolated_loader(tmp_path).discover_all()

    assert len(discovered) == 1
    assert discovered[0].manifest is not None
    assert discovered[0].manifest.name == "pyproject_plugin"
    assert discovered[0].plugin.metadata.name == "pyproject_plugin"


def test_malformed_plugin_json_is_skipped(tmp_path: Path):
    package_dir = tmp_path / "malformed_package"
    package_dir.mkdir()
    (package_dir / "plugin.json").write_text("{not valid json", encoding="utf-8")

    assert isolated_loader(tmp_path).discover_all() == []


def test_manifest_without_entry_point_is_ignored(tmp_path: Path):
    package_dir = tmp_path / "missing_entry_point_package"
    write_plugin_module(package_dir, "missing_entry_plugin", plugin_name="missing_entry_plugin")
    write_plugin_json(
        package_dir,
        name="missing_entry_plugin",
        module_name="missing_entry_plugin",
        extra={"entry_point": ""},
    )

    assert isolated_loader(tmp_path).discover_all() == []


def test_signed_external_plugin_json_loads_with_configured_trust_store(tmp_path: Path):
    package_dir = tmp_path / "signed_package"
    write_plugin_module(
        package_dir,
        "signed_plugin",
        class_name="SignedPlugin",
        plugin_name="signed_plugin",
        execute_result="signed-ok",
    )
    manifest_path = write_plugin_json(
        package_dir,
        name="signed_plugin",
        module_name="signed_plugin",
        class_name="SignedPlugin",
    )
    _sign_plugin_json(manifest_path)

    discovered = isolated_loader(
        tmp_path,
        plugin_trust_store_path=_write_trust_store(tmp_path),
    ).discover_all()

    assert len(discovered) == 1
    assert discovered[0].is_valid is True
    assert discovered[0].manifest is not None
    assert discovered[0].manifest.signature_key_id == "local-dev"
    assert discovered[0].plugin.execute() == "signed-ok"


def test_unsigned_external_plugin_json_is_rejected_when_trust_store_is_configured(tmp_path: Path):
    package_dir = tmp_path / "unsigned_package"
    write_plugin_module(package_dir, "unsigned_plugin", plugin_name="unsigned_plugin")
    write_plugin_json(package_dir, name="unsigned_plugin", module_name="unsigned_plugin")

    discovered = isolated_loader(
        tmp_path,
        plugin_trust_store_path=_write_trust_store(tmp_path),
    ).discover_all()

    assert len(discovered) == 1
    assert discovered[0].plugin is None
    assert discovered[0].is_valid is False
    assert "signature verification failed" in discovered[0].load_errors[0]


def test_tampered_external_plugin_json_is_rejected_before_load(tmp_path: Path):
    package_dir = tmp_path / "tampered_package"
    write_plugin_module(package_dir, "tampered_plugin", plugin_name="tampered_plugin")
    manifest_path = write_plugin_json(package_dir, name="tampered_plugin", module_name="tampered_plugin")
    _sign_plugin_json(manifest_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["version"] = "9.9.9"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    discovered = isolated_loader(
        tmp_path,
        plugin_trust_store_path=_write_trust_store(tmp_path),
    ).discover_all()

    assert len(discovered) == 1
    assert discovered[0].plugin is None
    assert discovered[0].is_valid is False
    assert "does not match trusted key" in discovered[0].load_errors[0]
