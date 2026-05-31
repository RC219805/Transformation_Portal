"""PluginLoader manifest discovery tests."""

from __future__ import annotations

import json
import logging
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


def test_malformed_plugin_json_does_not_fall_back_to_pyproject_under_trust_store(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    package_dir = tmp_path / "malformed_json_with_pyproject_package"
    write_plugin_module(package_dir, "fallback_plugin", plugin_name="fallback_plugin")
    (package_dir / "plugin.json").write_text("{not valid json", encoding="utf-8")
    write_pyproject_manifest(
        package_dir,
        name="fallback_plugin",
        module_name="fallback_plugin",
    )

    with caplog.at_level(logging.WARNING):
        discovered = isolated_loader(
            tmp_path,
            plugin_trust_store_path=_write_trust_store(tmp_path),
        ).discover_all()

    assert len(discovered) == 1
    assert discovered[0].plugin is None
    assert discovered[0].is_valid is False
    assert "Failed to parse" in caplog.text
    assert "signature verification failed" in discovered[0].load_errors[0]


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


def test_unsigned_external_plugin_json_is_rejected_before_module_import(tmp_path: Path):
    package_dir = tmp_path / "unsigned_import_guard_package"
    side_effect_path = tmp_path / "imported.txt"
    package_dir.mkdir()
    (package_dir / "unsigned_import_guard_plugin.py").write_text(
        f"""from __future__ import annotations

from pathlib import Path

from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

Path({str(side_effect_path)!r}).write_text("imported", encoding="utf-8")


class UnsignedImportGuardPlugin(PluginInterface):
    def _create_metadata(self):
        return PluginMetadata(
            name="unsigned_import_guard_plugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
            description="test plugin",
            author="tests",
        )

    def execute(self, *args, **kwargs):
        return "should-not-run"
""",
        encoding="utf-8",
    )
    write_plugin_json(
        package_dir,
        name="unsigned_import_guard_plugin",
        module_name="unsigned_import_guard_plugin",
        class_name="UnsignedImportGuardPlugin",
    )

    discovered = isolated_loader(
        tmp_path,
        plugin_trust_store_path=_write_trust_store(tmp_path),
    ).discover_all()

    assert len(discovered) == 1
    assert discovered[0].plugin is None
    assert discovered[0].is_valid is False
    assert not side_effect_path.exists()


def test_env_configured_trust_store_rejects_unsigned_plugin_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    package_dir = tmp_path / "env_unsigned_package"
    write_plugin_module(package_dir, "env_unsigned_plugin", plugin_name="env_unsigned_plugin")
    write_plugin_json(package_dir, name="env_unsigned_plugin", module_name="env_unsigned_plugin")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGIN_TRUST_STORE", str(_write_trust_store(tmp_path)))

    discovered = isolated_loader(tmp_path).discover_all()

    assert len(discovered) == 1
    assert discovered[0].plugin is None
    assert discovered[0].is_valid is False
    assert "signature verification failed" in discovered[0].load_errors[0]


def test_env_configured_trust_store_accepts_signed_plugin_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    package_dir = tmp_path / "env_signed_package"
    write_plugin_module(
        package_dir,
        "env_signed_plugin",
        class_name="EnvSignedPlugin",
        plugin_name="env_signed_plugin",
        execute_result="env-signed-ok",
    )
    manifest_path = write_plugin_json(
        package_dir,
        name="env_signed_plugin",
        module_name="env_signed_plugin",
        class_name="EnvSignedPlugin",
    )
    _sign_plugin_json(manifest_path)
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGIN_TRUST_STORE", str(_write_trust_store(tmp_path)))

    discovered = isolated_loader(tmp_path).discover_all()

    assert len(discovered) == 1
    assert discovered[0].is_valid is True
    assert discovered[0].manifest is not None
    assert discovered[0].manifest.signature_key_id == "local-dev"
    assert discovered[0].plugin.execute() == "env-signed-ok"


def test_explicit_trust_store_path_overrides_env_trust_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    package_dir = tmp_path / "explicit_trust_package"
    write_plugin_module(
        package_dir,
        "explicit_trust_plugin",
        class_name="ExplicitTrustPlugin",
        plugin_name="explicit_trust_plugin",
        execute_result="explicit-trust-ok",
    )
    manifest_path = write_plugin_json(
        package_dir,
        name="explicit_trust_plugin",
        module_name="explicit_trust_plugin",
        class_name="ExplicitTrustPlugin",
    )
    _sign_plugin_json(manifest_path, secret="explicit-secret")
    env_trust_store = tmp_path / "env-trust.json"
    env_trust_store.write_text(
        json.dumps({"keys": {"local-dev": "wrong-secret"}}, indent=2) + "\n",
        encoding="utf-8",
    )
    explicit_trust_store = tmp_path / "explicit-trust.json"
    explicit_trust_store.write_text(
        json.dumps({"keys": {"local-dev": "explicit-secret"}}, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRANSFORMATION_PORTAL_PLUGIN_TRUST_STORE", str(env_trust_store))

    discovered = isolated_loader(tmp_path, plugin_trust_store_path=explicit_trust_store).discover_all()

    assert len(discovered) == 1
    assert discovered[0].is_valid is True
    assert discovered[0].manifest is not None
    assert discovered[0].manifest.signature_key_id == "local-dev"
    assert discovered[0].plugin.execute() == "explicit-trust-ok"


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


def test_non_canonical_signed_plugin_json_is_rejected_before_module_import(tmp_path: Path):
    package_dir = tmp_path / "non_canonical_signed_package"
    side_effect_path = tmp_path / "imported.txt"
    package_dir.mkdir()
    (package_dir / "non_canonical_plugin.py").write_text(
        f"""from __future__ import annotations

from pathlib import Path

from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

Path({str(side_effect_path)!r}).write_text("imported", encoding="utf-8")


class NonCanonicalPlugin(PluginInterface):
    def _create_metadata(self):
        return PluginMetadata(
            name="non_canonical_plugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
            description="test plugin",
            author="tests",
        )

    def execute(self, *args, **kwargs):
        return "should-not-run"
""",
        encoding="utf-8",
    )
    (package_dir / "plugin.json").write_text(
        """{
  "name": "non_canonical_plugin",
  "version": "1.0.0",
  "plugin_type": "custom",
  "entry_point": "non_canonical_plugin:NonCanonicalPlugin",
  "quality_score": NaN,
  "signature_algorithm": "hmac-sha256",
  "signature_key_id": "local-dev",
  "signature": "not-a-valid-signature"
}
""",
        encoding="utf-8",
    )

    discovered = isolated_loader(
        tmp_path,
        plugin_trust_store_path=_write_trust_store(tmp_path),
    ).discover_all()

    assert len(discovered) == 1
    assert discovered[0].plugin is None
    assert discovered[0].is_valid is False
    assert not side_effect_path.exists()
    assert "signature verification failed" in discovered[0].load_errors[0]
    assert "Out of range" in discovered[0].load_errors[0]


def test_malformed_trust_store_rejects_signed_plugin_before_load(tmp_path: Path):
    package_dir = tmp_path / "bad_trust_store_package"
    write_plugin_module(package_dir, "bad_trust_plugin", plugin_name="bad_trust_plugin")
    manifest_path = write_plugin_json(package_dir, name="bad_trust_plugin", module_name="bad_trust_plugin")
    _sign_plugin_json(manifest_path)
    trust_store_path = tmp_path / "bad-trust-store.json"
    trust_store_path.write_text("{not valid json", encoding="utf-8")

    discovered = isolated_loader(
        tmp_path,
        plugin_trust_store_path=trust_store_path,
    ).discover_all()

    assert len(discovered) == 1
    assert discovered[0].plugin is None
    assert discovered[0].is_valid is False
    assert "signature verification failed" in discovered[0].load_errors[0]


def test_missing_trust_store_rejects_signed_plugin_before_load(tmp_path: Path):
    package_dir = tmp_path / "missing_trust_store_package"
    write_plugin_module(package_dir, "missing_trust_plugin", plugin_name="missing_trust_plugin")
    manifest_path = write_plugin_json(package_dir, name="missing_trust_plugin", module_name="missing_trust_plugin")
    _sign_plugin_json(manifest_path)

    discovered = isolated_loader(
        tmp_path,
        plugin_trust_store_path=tmp_path / "missing-trust-store.json",
    ).discover_all()

    assert len(discovered) == 1
    assert discovered[0].plugin is None
    assert discovered[0].is_valid is False
    assert "signature verification failed" in discovered[0].load_errors[0]
    assert "missing-trust-store.json" in discovered[0].load_errors[0]


def test_pyproject_only_external_plugin_is_rejected_when_trust_store_is_configured(tmp_path: Path):
    package_dir = tmp_path / "pyproject_only_package"
    write_plugin_module(package_dir, "pyproject_only_plugin", plugin_name="pyproject_only_plugin")
    write_pyproject_manifest(
        package_dir,
        name="pyproject_only_plugin",
        module_name="pyproject_only_plugin",
    )

    discovered = isolated_loader(
        tmp_path,
        plugin_trust_store_path=_write_trust_store(tmp_path),
    ).discover_all()

    assert len(discovered) == 1
    assert discovered[0].plugin is None
    assert discovered[0].is_valid is False
    assert discovered[0].load_errors == [
        "External plugin packages require a signed plugin.json manifest when plugin trust is configured"
    ]


def test_single_file_external_plugin_is_skipped_when_trust_store_is_configured(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    plugin_path = write_plugin_module(
        tmp_path,
        "single_file_plugin",
        plugin_name="single_file_plugin",
    )

    with caplog.at_level(logging.WARNING):
        discovered = isolated_loader(
            tmp_path,
            plugin_trust_store_path=_write_trust_store(tmp_path),
        ).discover_all()

    assert discovered == []
    assert "requires signed plugin.json manifests" in caplog.text
    assert str(plugin_path) in caplog.text


def test_trust_store_signature_requirement_keeps_builtin_plugins_exempt(tmp_path: Path):
    loader = isolated_loader(
        tmp_path,
        plugin_trust_store_path=_write_trust_store(tmp_path),
    )

    builtin_plugin = loader._builtin_plugins_root() / "processors.py"  # noqa: SLF001 - tests pin trust boundary.
    external_plugin = tmp_path / "external_plugin.py"

    assert loader._requires_manifest_signature(external_plugin) is True  # noqa: SLF001
    assert loader._requires_manifest_signature(builtin_plugin) is False  # noqa: SLF001
