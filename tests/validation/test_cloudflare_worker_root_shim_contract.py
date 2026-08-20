from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_ROOT = PROJECT_ROOT / "cloudflare" / "transformationportal-worker"
PARITY_TOOL_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_worker_dependency_parity.py"
PARITY_SPEC = importlib.util.spec_from_file_location("check_worker_dependency_parity", PARITY_TOOL_PATH)
assert PARITY_SPEC is not None and PARITY_SPEC.loader is not None
parity = importlib.util.module_from_spec(PARITY_SPEC)
PARITY_SPEC.loader.exec_module(parity)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dependency_surfaces() -> dict[str, dict]:
    return {
        "root_package": _load_json(PROJECT_ROOT / "package.json"),
        "root_lock": _load_json(PROJECT_ROOT / "package-lock.json"),
        "worker_package": _load_json(WORKER_ROOT / "package.json"),
        "worker_lock": _load_json(WORKER_ROOT / "package-lock.json"),
    }


def _set_nested(mapping: dict, path: tuple[str, ...], value: object) -> None:
    current = mapping
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def test_root_worker_dependency_parity_validator_passes_current_surfaces() -> None:
    surfaces = _dependency_surfaces()

    assert (
        parity.validate_worker_dependency_parity(
            surfaces["root_package"],
            surfaces["root_lock"],
            surfaces["worker_package"],
            surfaces["worker_lock"],
        )
        == []
    )


@pytest.mark.parametrize(
    ("version", "constraint", "expected"),
    (
        ("1.9.0", "^1.2.3", True),
        ("2.0.0", "^1.2.3", False),
        ("0.2.99", "^0.2.3", True),
        ("0.3.0", "^0.2.3", False),
        ("0.99.0", "^0.2.3", False),
        ("0.0.3", "^0.0.3", True),
        ("0.0.4", "^0.0.3", False),
        ("0.0.0", "^0.0.0", True),
        ("0.0.1", "^0.0.0", False),
        ("5.20260815.1", "^5.20260811.1", True),
        ("6.0.0", "^5.20260811.1", False),
        ("5.20260815.1.0", "^5.20260811.1", False),
        ("05.20260815.1", "^5.20260811.1", False),
    ),
)
def test_worker_dependency_parity_applies_numeric_caret_semantics(
    version: str,
    constraint: str,
    expected: bool,
) -> None:
    assert parity._version_satisfies_caret(version, constraint) is expected


@pytest.mark.parametrize(
    "wrangler_spec",
    (
        "^4.123.0",
        "4.123.0-beta.1",
        "4.123.0+metadata",
        "4.123.0.1",
        "04.123.0",
        "4.0123.0",
        "4.123.00",
        "npm:wrangler@4.123.0",
    ),
)
def test_root_worker_dependency_parity_requires_stable_numeric_wrangler_pins(
    wrangler_spec: str,
) -> None:
    surfaces = copy.deepcopy(_dependency_surfaces())
    surfaces["root_package"]["devDependencies"]["wrangler"] = wrangler_spec
    surfaces["worker_package"]["devDependencies"]["wrangler"] = wrangler_spec

    errors = parity.validate_worker_dependency_parity(
        surfaces["root_package"],
        surfaces["root_lock"],
        surfaces["worker_package"],
        surfaces["worker_lock"],
    )

    assert errors[:2] == [
        "root manifest must exact-pin Wrangler to a stable numeric release",
        "Worker manifest must exact-pin Wrangler to a stable numeric release",
    ]


def test_root_worker_dependency_parity_reports_lock_field_differences_in_stable_order() -> None:
    surfaces = copy.deepcopy(_dependency_surfaces())
    root_wrangler = surfaces["root_lock"]["packages"]["node_modules/wrangler"]
    root_wrangler["integrity"] = "sha512-mutated"
    root_wrangler["engines"] = {"node": ">=999"}

    errors = parity.validate_worker_dependency_parity(
        surfaces["root_package"],
        surfaces["root_lock"],
        surfaces["worker_package"],
        surfaces["worker_lock"],
    )

    assert errors == [
        "root and Worker Wrangler lock entries must match field 'integrity'",
        "root and Worker Wrangler lock entries must match field 'engines'",
    ]


def test_root_worker_dependency_parity_rejects_transitive_lock_divergence() -> None:
    surfaces = copy.deepcopy(_dependency_surfaces())
    transitive_path = "node_modules/@esbuild/darwin-arm64"
    surfaces["root_lock"]["packages"][transitive_path]["version"] = "0.0.0"

    errors = parity.validate_worker_dependency_parity(
        surfaces["root_package"],
        surfaces["root_lock"],
        surfaces["worker_package"],
        surfaces["worker_lock"],
    )

    assert errors == [f"root and Worker shared toolchain lock entries must match for {transitive_path!r}"]


def test_root_worker_dependency_parity_rejects_shared_graph_path_drift() -> None:
    surfaces = copy.deepcopy(_dependency_surfaces())
    transitive_path = "node_modules/@esbuild/darwin-arm64"
    surfaces["worker_lock"]["packages"].pop(transitive_path)

    errors = parity.validate_worker_dependency_parity(
        surfaces["root_package"],
        surfaces["root_lock"],
        surfaces["worker_package"],
        surfaces["worker_lock"],
    )

    assert errors == [
        "root and Worker shared toolchain lock paths must match "
        f"(missing from Worker: [{transitive_path!r}]; extra in Worker: [])"
    ]


@pytest.mark.parametrize(
    ("surface", "path", "value", "expected_error"),
    (
        (
            "root_package",
            ("devDependencies", "wrangler"),
            "4.999.0",
            "root and Worker manifests must exact-pin the same Wrangler version",
        ),
        (
            "root_lock",
            ("packages", "", "devDependencies", "wrangler"),
            "4.999.0",
            "root lockfile manifest entry must match its Wrangler manifest pin",
        ),
        (
            "root_lock",
            ("packages", "node_modules/wrangler", "integrity"),
            "sha512-mutated",
            "root and Worker Wrangler lock entries must match field 'integrity'",
        ),
        (
            "worker_package",
            ("devDependencies", "@cloudflare/workers-types"),
            "5.20250101.1",
            "Worker lockfile manifest entry must match its @cloudflare/workers-types manifest pin",
        ),
        (
            "worker_lock",
            ("packages", "node_modules/@cloudflare/workers-types", "version"),
            "5.20250101.1",
            "Worker lockfile must resolve its exact @cloudflare/workers-types manifest pin",
        ),
    ),
)
def test_root_worker_dependency_parity_rejects_manifest_and_lock_mutations(
    surface: str,
    path: tuple[str, ...],
    value: object,
    expected_error: str,
) -> None:
    surfaces = copy.deepcopy(_dependency_surfaces())
    _set_nested(surfaces[surface], path, value)

    errors = parity.validate_worker_dependency_parity(
        surfaces["root_package"],
        surfaces["root_lock"],
        surfaces["worker_package"],
        surfaces["worker_lock"],
    )

    assert expected_error in errors


def test_root_worker_dependency_parity_rejects_removed_worker_types_dependency() -> None:
    surfaces = copy.deepcopy(_dependency_surfaces())
    surfaces["worker_package"]["devDependencies"].pop("@cloudflare/workers-types")
    surfaces["worker_lock"]["packages"][""]["devDependencies"].pop("@cloudflare/workers-types")
    surfaces["worker_lock"]["packages"].pop("node_modules/@cloudflare/workers-types")

    errors = parity.validate_worker_dependency_parity(
        surfaces["root_package"],
        surfaces["root_lock"],
        surfaces["worker_package"],
        surfaces["worker_lock"],
    )

    assert "Worker manifest must exact-pin @cloudflare/workers-types to a stable numeric release" in errors
    assert "Worker @cloudflare/workers-types pin must satisfy Wrangler's peer range" in errors


def test_root_worker_build_package_is_minimal_deploy_shim() -> None:
    """The root package exists only for Cloudflare Workers Builds."""
    root_package = _load_json(PROJECT_ROOT / "package.json")
    worker_package = _load_json(WORKER_ROOT / "package.json")

    assert root_package["name"] == "transformation-portal-worker-build"
    assert root_package["private"] is True
    assert root_package["scripts"] == {
        "worker:dry-run": worker_package["scripts"]["dry-run"],
        "worker:deploy": worker_package["scripts"]["deploy"],
    }
    assert root_package["scripts"]["worker:deploy"].endswith(" --keep-vars")
    assert root_package["engines"] == worker_package["engines"]
    assert root_package["packageManager"] == worker_package["packageManager"]
    assert root_package["overrides"] == worker_package["overrides"]
    assert root_package["devDependencies"] == {
        "wrangler": worker_package["devDependencies"]["wrangler"],
    }
    assert "dependencies" not in root_package


def test_root_worker_build_lock_matches_package_contract() -> None:
    """The root lockfile must stay in sync with the minimal build package."""
    root_package = _load_json(PROJECT_ROOT / "package.json")
    root_lock = _load_json(PROJECT_ROOT / "package-lock.json")
    root_lock_package = root_lock["packages"][""]
    wrangler_spec = root_package["devDependencies"]["wrangler"]

    assert root_lock["name"] == root_package["name"]
    assert root_lock["lockfileVersion"] == 3
    assert root_lock["requires"] is True
    assert root_lock_package["name"] == root_package["name"]
    assert root_lock_package["devDependencies"] == root_package["devDependencies"]
    assert root_lock_package["engines"] == root_package["engines"]
    assert "dependencies" not in root_lock_package
    assert root_lock["packages"]["node_modules/wrangler"]["version"] == wrangler_spec
    assert root_lock["packages"]["node_modules/wrangler"]["bin"] == {
        "cf-wrangler": "bin/cf-wrangler.js",
        "wrangler": "bin/wrangler.js",
        "wrangler2": "bin/wrangler.js",
    }


def test_root_and_worker_locks_apply_the_same_security_overrides() -> None:
    """Worker build surfaces must resolve governed transitive fixes identically."""
    root_package = _load_json(PROJECT_ROOT / "package.json")
    worker_package = _load_json(WORKER_ROOT / "package.json")
    root_lock = _load_json(PROJECT_ROOT / "package-lock.json")
    worker_lock = _load_json(WORKER_ROOT / "package-lock.json")

    assert root_package["overrides"] == worker_package["overrides"]
    for dependency, version in root_package["overrides"].items():
        lock_key = f"node_modules/{dependency}"
        assert root_lock["packages"][lock_key]["version"] == version
        assert worker_lock["packages"][lock_key]["version"] == version


def test_root_wrangler_config_points_to_governed_worker_entrypoint() -> None:
    """The root Worker config should delegate to the governed Worker package."""
    root_config = _load_json(PROJECT_ROOT / "wrangler.jsonc")
    worker_config = _load_json(WORKER_ROOT / "wrangler.jsonc")

    expected_root_main = f"cloudflare/transformationportal-worker/{worker_config['main']}"
    worker_entrypoint = PROJECT_ROOT / expected_root_main
    entrypoint_source = worker_entrypoint.read_text(encoding="utf-8")

    assert root_config["$schema"] == "./node_modules/wrangler/config-schema.json"
    assert root_config["name"] == worker_config["name"] == "transformationportal"
    assert root_config["main"] == expected_root_main
    assert worker_entrypoint.is_file()
    assert root_config["compatibility_date"] == worker_config["compatibility_date"]
    assert root_config["keep_vars"] is worker_config["keep_vars"] is True
    assert root_config["observability"] == worker_config["observability"]
    assert root_config["observability"]["enabled"] is True
    assert 0 < root_config["observability"]["head_sampling_rate"] <= 1
    assert "export default" in entrypoint_source
    assert "async fetch(" in entrypoint_source
    assert "FRONTDOOR_ORIGIN" in entrypoint_source
