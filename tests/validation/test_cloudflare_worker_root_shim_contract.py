from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_ROOT = PROJECT_ROOT / "cloudflare" / "transformationportal-worker"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    assert root_package["engines"] == worker_package["engines"]
    assert root_package["packageManager"] == worker_package["packageManager"]
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
        "wrangler": "bin/wrangler.js",
        "wrangler2": "bin/wrangler.js",
    }


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
    assert root_config["observability"] == worker_config["observability"]
    assert root_config["observability"]["enabled"] is True
    assert 0 < root_config["observability"]["head_sampling_rate"] <= 1
    assert "export default" in entrypoint_source
    assert "async fetch(" in entrypoint_source
    assert "FRONTDOOR_ORIGIN" in entrypoint_source
