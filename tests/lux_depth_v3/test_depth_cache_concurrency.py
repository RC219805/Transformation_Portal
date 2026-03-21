from __future__ import annotations

import logging
import threading

import numpy as np

from transformation_portal.lux_depth_v3.depth_cache import DepthCache
import pytest



pytestmark = pytest.mark.unit

def test_depth_cache_concurrent_same_key_does_not_emit_store_failures(tmp_path, caplog):
    """Concurrent same-key writes should not produce internal store-failure warnings."""
    cache = DepthCache(tmp_path, max_size_gb=1.0)

    def store_depth(value: int) -> None:
        depth = np.full((100, 100), value, dtype=np.float32)
        cache.store("same_image", "same_config", depth)

    with caplog.at_level(logging.WARNING, logger="transformation_portal.lux_depth_v3.depth_cache"):
        threads = [threading.Thread(target=store_depth, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert not any("Failed to cache depth" in record.message for record in caplog.records)

    cached = cache.get("same_image", "same_config")
    assert cached is not None
