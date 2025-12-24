from pathlib import Path

import numpy as np

from lux_depth_v2.depth_cache_manager import DepthCacheManager


def test_depth_cache_manager_roundtrip(tmp_path: Path):
    mgr = DepthCacheManager(tmp_path / "depth_cache")
    depth = np.random.rand(64, 64).astype(np.float32)
    mgr.save("k", depth, {"model": "v2-large"}, 0.85)

    loaded = mgr.load("k")
    assert loaded is not None
    assert loaded["depth"].shape == (64, 64)
    assert abs(float(loaded["confidence_proxy"]) - 0.85) < 1e-6
