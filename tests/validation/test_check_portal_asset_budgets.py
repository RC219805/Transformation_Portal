"""Unit tests for the portal asset budget validator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [
    pytest.mark.unit,
]


def _load_budget_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "validation" / "check_portal_asset_budgets.py"
    spec = importlib.util.spec_from_file_location(
        "check_portal_asset_budgets_under_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def budget_module() -> ModuleType:
    return _load_budget_module()


def test_repo_budget_contract_covers_shared_ui_tokens() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = json.loads((repo_root / "config" / "portal_asset_budgets.json").read_text(encoding="utf-8"))
    assert payload["assets"]["shared-ui-tokens.css"] == {
        "max_bytes": 2048,
        "max_gzip_bytes": 768,
    }


def test_read_budgets_rejects_non_object_asset_budget(
    budget_module: ModuleType,
    tmp_path: Path,
) -> None:
    budget_path = tmp_path / "portal_asset_budgets.json"
    budget_path.write_text(
        json.dumps(
            {
                "schema": "tp.portal_asset_budgets.v1",
                "assets": {
                    "portal.js": 123,
                },
            }
        ),
        encoding="utf-8",
    )
    budget_module.BUDGET_PATH = budget_path

    with pytest.raises(
        RuntimeError,
        match="Portal asset budget for portal.js must be an object",
    ):
        budget_module._read_budgets()


def test_read_budgets_rejects_non_positive_integer_limit(
    budget_module: ModuleType,
    tmp_path: Path,
) -> None:
    budget_path = tmp_path / "portal_asset_budgets.json"
    budget_path.write_text(
        json.dumps(
            {
                "schema": "tp.portal_asset_budgets.v1",
                "assets": {
                    "portal.js": {
                        "max_bytes": 0,
                        "max_gzip_bytes": 10,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    budget_module.BUDGET_PATH = budget_path

    with pytest.raises(
        RuntimeError,
        match=("Portal asset budget for portal.js must define a positive integer " "max_bytes"),
    ):
        budget_module._read_budgets()


def test_main_validates_budgeted_assets(
    budget_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assets_dir = tmp_path / "public" / "portal-assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for asset_name, content in {
        "portal.js": "console.log('ok');\n",
        "portal.css": "body { color: black; }\n",
        "shared-ui-tokens.css": ":root { --surface: #111; }\n",
    }.items():
        (assets_dir / asset_name).write_text(content, encoding="utf-8")

    budget_path = tmp_path / "portal_asset_budgets.json"
    budget_path.write_text(
        json.dumps(
            {
                "schema": "tp.portal_asset_budgets.v1",
                "assets": {
                    "portal.js": {
                        "max_bytes": 1024,
                        "max_gzip_bytes": 256,
                    },
                    "portal.css": {
                        "max_bytes": 1024,
                        "max_gzip_bytes": 256,
                    },
                    "shared-ui-tokens.css": {
                        "max_bytes": 1024,
                        "max_gzip_bytes": 256,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    budget_module.BUDGET_PATH = budget_path
    budget_module.PORTAL_ASSETS_DIR = assets_dir

    assert budget_module.main() == 0
    captured = capsys.readouterr()
    assert "shared-ui-tokens.css: raw=" in captured.out
    assert "portal asset budgets: OK" in captured.out
