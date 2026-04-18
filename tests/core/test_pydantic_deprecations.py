from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest
from pydantic.warnings import PydanticDeprecatedSince20

pytestmark = pytest.mark.unit


def test_core_config_and_storage_emit_no_pydantic_v2_deprecations() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=PydanticDeprecatedSince20)

        schemas = importlib.import_module("transformation_portal.core.config.schemas")
        export_manager = importlib.import_module("transformation_portal.core.storage.export_manager")

        importlib.reload(schemas)
        importlib.reload(export_manager)

        schemas.PerformanceConfig()
        export_manager.autotune_export_config(np.zeros((2, 2, 3), dtype=np.uint8), export_manager.ExportConfig())
