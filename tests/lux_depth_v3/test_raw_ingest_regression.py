"""Regression guard: RAW files must route through canonical ingest decoder."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from transformation_portal.lux_depth_v3.preprocessing import preprocess_image
import pytest



pytestmark = pytest.mark.unit

def test_raw_ingest_uses_canonical_decoder(tmp_path: Path) -> None:
    raw = tmp_path / "test.DNG"
    raw.write_bytes(b"phase_c1_raw_payload_16b")

    config = SimpleNamespace(raw_ingest_mode="auto", raw_wb_mode="camera", raw_demosaic="AHD")

    with (
        patch("transformation_portal.spatial_ai.ingest.contracts.decode_contract") as decoder,
        patch("transformation_portal.lux_depth_v3.preprocessing.Image.open") as pil_open,
    ):
        decoder.return_value = np.full((20, 20, 3), 0.5, dtype=np.float32)
        preprocess_image(raw, raw_config=config)

    assert decoder.called
    assert not pil_open.called
