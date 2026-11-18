"""Regression tests for the depth-map ``DepthCache`` (audit finding D-09)."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from transformation_portal.core.security import serialization
from transformation_portal.depth.utils import cache as cache_module
from transformation_portal.depth.utils.cache import DepthCache

pytestmark = [pytest.mark.unit, pytest.mark.regression]

_NAMESPACE = "da2-float-prediction-v2"


def _result(tag: float, shape) -> dict:
    depth = np.full(shape, tag, dtype=np.float32)
    return {"depth": depth, "depth_raw": depth * 10.0, "metadata": {"tag": tag, "shape": tuple(shape)}}


def test_namespaced_key_distinguishes_shape_and_dtype():
    cache = DepthCache(max_size=8, namespace=_NAMESPACE)
    landscape = np.zeros((40, 60, 3), dtype=np.uint8)
    portrait = np.zeros((60, 40, 3), dtype=np.uint8)

    cache.get_or_compute(landscape, lambda: _result(1.0, (40, 60)))
    got = cache.get_or_compute(portrait, lambda: _result(2.0, (60, 40)))

    assert got["depth"].shape == (60, 40)
    assert cache._generate_key(landscape) != cache._generate_key(portrait)
    same_bytes_other_dtype = np.zeros((40, 60, 3), dtype=np.int8)
    assert cache._generate_key(landscape) != cache._generate_key(same_bytes_other_dtype)


def test_namespaced_key_distinguishes_sub_quantum_float_images():
    cache = DepthCache(max_size=8, namespace=_NAMESPACE)
    image = np.linspace(0.0, 1.0, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    nudged = image.copy()
    nudged[0, 0, 1] += np.float32(1e-4)  # below one 8-bit step after min-max normalisation

    assert cache._generate_key(image) != cache._generate_key(nudged)
    assert cache._generate_key(image) == cache._generate_key(image.copy())
    assert cache._generate_key(image) == cache._generate_key(np.asfortranarray(image))


def test_legacy_key_is_unchanged_without_namespace():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert DepthCache()._generate_key(image) == hashlib.md5(image.tobytes(), usedforsecurity=False).hexdigest()


def test_callers_cannot_mutate_the_cached_entry():
    cache = DepthCache(max_size=8, namespace=_NAMESPACE)
    image = np.full((8, 8, 3), 0.25, dtype=np.float32)

    computed = _result(0.5, (8, 8))
    first = cache.get_or_compute(image, lambda: computed)
    assert first is computed
    first["depth"] *= 0.0
    first["metadata"]["tag"] = "mutated"

    second = cache.get_or_compute(image, lambda: pytest.fail("expected a memory hit"))
    assert float(second["depth"].max()) == 0.5
    assert second["metadata"]["tag"] == 0.5
    second["depth"][:] = -1.0

    third = cache.get_or_compute(image, lambda: pytest.fail("expected a memory hit"))
    assert float(third["depth"].min()) == 0.5


def test_disk_hit_returns_a_private_copy(tmp_path):
    image = np.full((8, 8, 3), 7, dtype=np.uint8)
    DepthCache(cache_dir=tmp_path, enable_disk_cache=True, namespace=_NAMESPACE).put(image, _result(0.5, (8, 8)))

    reopened = DepthCache(cache_dir=tmp_path, enable_disk_cache=True, namespace=_NAMESPACE)
    first = reopened.get_or_compute(image, lambda: pytest.fail("expected a disk hit"))
    first["depth"][:] = 0.0
    second = reopened.get_or_compute(image, lambda: pytest.fail("expected a memory hit"))

    assert float(second["depth"].max()) == 0.5


def test_disk_write_is_atomic_and_leaves_no_temp_files(tmp_path, monkeypatch):
    image = np.full((8, 8, 3), 3, dtype=np.uint8)
    cache = DepthCache(cache_dir=tmp_path, enable_disk_cache=True, namespace=_NAMESPACE)
    key = cache._generate_key(image)
    target = tmp_path / f"{key}.pkl"

    cache.put(image, _result(0.5, (8, 8)))
    good_bytes = target.read_bytes()

    def interrupted_dump(obj, fh, protocol=None):
        fh.write(b"partial pickle bytes " * 8)
        raise OSError("disk full")

    monkeypatch.setattr(cache_module.pickle, "dump", interrupted_dump)
    cache.put(image, _result(0.75, (8, 8)))

    assert target.read_bytes() == good_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == [target.name]
    monkeypatch.undo()

    reopened = DepthCache(cache_dir=tmp_path, enable_disk_cache=True, namespace=_NAMESPACE)
    assert float(reopened.get_or_compute(image, lambda: pytest.fail("expected a disk hit"))["depth"].max()) == 0.5


def test_disk_hit_unpickles_once(tmp_path, monkeypatch):
    image = np.full((8, 8, 3), 9, dtype=np.uint8)
    DepthCache(cache_dir=tmp_path, enable_disk_cache=True, namespace=_NAMESPACE).put(image, _result(0.5, (8, 8)))

    calls = []
    real_load = serialization.safe_pickle_load

    def counting_load(fh, *args, **kwargs):
        calls.append(True)
        return real_load(fh, *args, **kwargs)

    monkeypatch.setattr(cache_module, "safe_pickle_load", counting_load)
    reopened = DepthCache(cache_dir=tmp_path, enable_disk_cache=True, namespace=_NAMESPACE)
    got = reopened.get_or_compute(image, lambda: pytest.fail("expected a disk hit"))

    assert float(got["depth"].max()) == 0.5
    assert len(calls) == 1


def test_disk_round_trip_keeps_values_beyond_fp16_range(tmp_path):
    image = np.full((4, 4, 3), 5, dtype=np.uint8)
    depth_raw = np.full((4, 4), 1.0e5, dtype=np.float32)  # e.g. metric depth in millimetres
    DepthCache(cache_dir=tmp_path, enable_disk_cache=True, namespace=_NAMESPACE).put(
        image, {"depth": np.full((4, 4), 0.5, dtype=np.float32), "depth_raw": depth_raw}
    )

    reopened = DepthCache(cache_dir=tmp_path, enable_disk_cache=True, namespace=_NAMESPACE)
    got = reopened.get_or_compute(image, lambda: pytest.fail("expected a disk hit"))

    assert np.isfinite(got["depth_raw"]).all()
    np.testing.assert_array_equal(got["depth_raw"], depth_raw)
    assert got["depth_raw"].dtype == np.float32
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))
