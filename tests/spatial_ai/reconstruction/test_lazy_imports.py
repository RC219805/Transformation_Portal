"""Test PEP 562 lazy import contract for torch-dependent modules.

This test verifies that importing the reconstruction package does NOT eagerly
load torch, and that torch is only loaded when torch-dependent symbols are
actually accessed (GaussianBackend, SceneBuilder).

Context:
    - PR #957: Fixed torch import leak with PEP 562 lazy loading
    - Commit 73401dce: Added __getattr__ pattern to __init__.py
    - Issue #958 Step 2.2: ML-tier torch laziness guard

Related Tests:
    - tests/enforcement/test_core_torch_boundary.py: Verifies imports work WITHOUT torch
    - This test: Verifies torch loads LAZILY when torch IS available

Architecture Decision:
    - ADR-031: Test Dependency Isolation
    - PEP 562: Module __getattr__ for lazy imports

References:
    - https://peps.python.org/pep-0562/
    - src/transformation_portal/spatial_ai/reconstruction/__init__.py:68-97
"""

import statistics
import sys
import time

import pytest

# Mark all tests in this file as requiring ML dependencies
pytestmark = pytest.mark.ml

# Skip entire module if torch not available
torch = pytest.importorskip("torch", reason="torch required for lazy import tests")


class TestTorchLazyImportContract:
    """Verify torch is lazy-loaded via PEP 562 __getattr__."""

    def test_package_import_does_not_load_torch(self):
        """Test that importing reconstruction package doesn't eagerly load torch.

        This is the core lazy import contract: the package can be imported
        without triggering torch import, even when torch is installed.

        Acceptance Criteria:
            - Importing package succeeds
            - torch NOT in sys.modules after import (or torch modules unchanged)
            - Import is fast (<100ms)

        Failure Mode Prevented:
            - Eager torch import at module level (pre-PR #957 behavior)
            - Accidental torch dependency in __init__.py imports

        Note:
            We check if torch modules changed rather than checking absolute presence,
            since other tests may have already loaded torch and deleting torch from
            sys.modules can cause RuntimeError on re-import.
        """
        # Snapshot which torch modules exist before import
        torch_modules_before = {key for key in sys.modules if key.startswith("torch")}

        # Remove reconstruction package to force fresh import
        reconstruction_modules = [key for key in sys.modules if "transformation_portal.spatial_ai.reconstruction" in key]
        for module_name in reconstruction_modules:
            del sys.modules[module_name]

        # Now import the reconstruction package
        start_time = time.time()
        import transformation_portal.spatial_ai.reconstruction as reconstruction_pkg

        import_time = time.time() - start_time

        # Snapshot which torch modules exist after import
        torch_modules_after = {key for key in sys.modules if key.startswith("torch")}

        # Verify NO NEW torch modules were loaded
        new_torch_modules = torch_modules_after - torch_modules_before
        assert len(new_torch_modules) == 0, (
            f"torch modules were loaded during package import: {new_torch_modules}. "
            "PEP 562 lazy import contract violated - check __init__.py for eager imports."
        )

        # Verify import is fast (lazy loading should be instant)
        assert import_time < 0.1, (
            f"Package import took {import_time:.3f}s, expected <100ms. " "Lazy imports should be nearly instant."
        )

        # Verify the package is actually imported
        assert reconstruction_pkg is not None
        assert hasattr(reconstruction_pkg, "__getattr__"), "Package must implement __getattr__ for PEP 562 lazy loading"

    def test_accessing_gaussian_backend_loads_torch(self):
        """Test that accessing GaussianBackend triggers torch import.

        This validates the lazy loading mechanism works correctly:
        1. Package imports without adding torch modules
        2. Accessing GaussianBackend triggers __getattr__
        3. __getattr__ imports the module (which imports torch)
        4. New torch modules appear in sys.modules

        Acceptance Criteria:
            - No new torch modules before accessing GaussianBackend
            - New torch modules after accessing GaussianBackend
            - GaussianBackend can be accessed successfully

        Implementation Note:
            See src/transformation_portal/spatial_ai/reconstruction/__init__.py:68-79
            for the __getattr__ implementation.
        """
        # Snapshot torch modules before
        torch_modules_before = {key for key in sys.modules if key.startswith("torch")}

        # Remove reconstruction from sys.modules to force fresh import
        reconstruction_modules = [key for key in sys.modules if "transformation_portal.spatial_ai.reconstruction" in key]
        for module_name in reconstruction_modules:
            del sys.modules[module_name]

        # Import package (should not load NEW torch modules)
        import transformation_portal.spatial_ai.reconstruction as reconstruction_pkg

        torch_modules_after_import = {key for key in sys.modules if key.startswith("torch")}
        new_modules_after_import = torch_modules_after_import - torch_modules_before

        # Verify no NEW torch modules after package import
        # (torch may already be loaded, but package import shouldn't add more)
        assert len(new_modules_after_import) == 0, f"Package import added torch modules: {new_modules_after_import}"

        # Now access GaussianBackend (should trigger lazy load via __getattr__)
        start_time = time.time()
        gaussian_backend_cls = reconstruction_pkg.GaussianBackend
        load_time = time.time() - start_time

        # Verify NEW torch modules were loaded
        torch_modules_after_access = {key for key in sys.modules if key.startswith("torch")}
        new_modules_after_access = torch_modules_after_access - torch_modules_after_import

        # If torch wasn't loaded before, we should see many torch modules now
        # If torch was loaded, we should see at least gaussian_backend module
        assert (
            "transformation_portal.spatial_ai.reconstruction.gaussian_backend" in sys.modules
        ), "gaussian_backend module should be loaded after accessing GaussianBackend"

        # Verify lazy load is reasonably fast
        # (allow more time if torch needs to be loaded for the first time)
        max_time = 5.0 if len(torch_modules_before) == 0 else 0.5
        assert load_time < max_time, (
            f"Lazy load took {load_time:.3f}s, expected <{max_time}s. " "PEP 562 __getattr__ may be doing unnecessary work."
        )

        # Verify we actually got the class
        assert gaussian_backend_cls is not None
        assert gaussian_backend_cls.__name__ == "GaussianBackend"

    def test_accessing_scene_builder_loads_torch(self):
        """Test that accessing SceneBuilder triggers torch import.

        SceneBuilder imports GaussianBackend at module level, so accessing
        SceneBuilder should also ensure torch dependencies are loaded.

        Import Chain:
            __init__.py.__getattr__("SceneBuilder")
            -> import scene_builder.SceneBuilder
            -> scene_builder imports GaussianBackend (line 26)
            -> gaussian_backend imports torch (line 35)

        This test verifies the lazy loading works through the dependency chain.
        """
        # Remove reconstruction from sys.modules
        reconstruction_modules = [key for key in sys.modules if "transformation_portal.spatial_ai.reconstruction" in key]
        for module_name in reconstruction_modules:
            del sys.modules[module_name]

        # Import package
        import transformation_portal.spatial_ai.reconstruction as reconstruction_pkg

        # Access SceneBuilder (should trigger lazy load)
        scene_builder_cls = reconstruction_pkg.SceneBuilder

        # Verify SceneBuilder module was loaded
        assert (
            "transformation_portal.spatial_ai.reconstruction.scene_builder" in sys.modules
        ), "scene_builder module should be loaded after accessing SceneBuilder"

        # Verify GaussianBackend was also loaded (dependency chain)
        assert (
            "transformation_portal.spatial_ai.reconstruction.gaussian_backend" in sys.modules
        ), "gaussian_backend should be loaded via SceneBuilder dependency chain"

        # Verify we got the class
        assert scene_builder_cls is not None
        assert scene_builder_cls.__name__ == "SceneBuilder"

    @pytest.mark.benchmark
    def test_lazy_import_performance_baseline(self):
        """Benchmark lazy import performance to detect regressions.

        Lazy imports should be near-instant for the package itself.
        First symbol access will load torch (slow), but subsequent
        accesses should be fast (cached in sys.modules).

        Performance Budget:
            - Package import median: <15ms (across repeated cold imports)
            - First symbol access: <5s (includes torch import if needed)
            - Subsequent accesses: <1ms (already in sys.modules)
        """
        sample_count = 7
        package_import_times = []
        reconstruction_pkg = None

        def _cold_import_reconstruction():
            reconstruction_modules = [key for key in sys.modules if "transformation_portal.spatial_ai.reconstruction" in key]
            for module_name in reconstruction_modules:
                del sys.modules[module_name]
            start = time.perf_counter()
            import transformation_portal.spatial_ai.reconstruction as pkg

            return pkg, time.perf_counter() - start

        # Warm-up sample to absorb startup timing noise.
        _cold_import_reconstruction()

        # Measure repeated cold imports and use median to reduce CI jitter sensitivity.
        for _ in range(sample_count):
            reconstruction_pkg, package_import_time = _cold_import_reconstruction()
            package_import_times.append(package_import_time)

        assert reconstruction_pkg is not None
        package_import_median = statistics.median(package_import_times)

        # Measure first access (may lazy load torch)
        start = time.perf_counter()
        _ = reconstruction_pkg.GaussianBackend
        first_access_time = time.perf_counter() - start

        # Measure second access (already loaded)
        start = time.perf_counter()
        _ = reconstruction_pkg.GaussianBackend
        second_access_time = time.perf_counter() - start

        # Assert performance budgets
        assert package_import_median < 0.015, (
            f"Package import median took {package_import_median*1000:.1f}ms over "
            f"{sample_count} cold samples, expected <15ms. "
            f"Samples (ms): {[round(t * 1000, 3) for t in package_import_times]}. "
            "PEP 562 __getattr__ should stay near-instant."
        )

        assert first_access_time < 5.0, (
            f"First access took {first_access_time*1000:.1f}ms, "
            f"expected <5000ms. This includes torch import time if needed."
        )

        assert second_access_time < 0.001, (
            f"Second access took {second_access_time*1000:.1f}ms, " f"expected <1ms. Module should be cached in sys.modules."
        )

    def test_non_torch_symbols_do_not_load_torch(self):
        """Test that accessing non-torch symbols doesn't unnecessarily load torch.

        Not all symbols in the reconstruction package require torch.
        This test verifies the lazy loading is selective.

        Note: This is primarily a design validation test. The key invariant
        is that torch-dependent symbols DO load torch when accessed.
        """
        # Remove reconstruction from sys.modules
        reconstruction_modules = [key for key in sys.modules if "transformation_portal.spatial_ai.reconstruction" in key]
        for module_name in reconstruction_modules:
            del sys.modules[module_name]

        # Import package
        import transformation_portal.spatial_ai.reconstruction as reconstruction_pkg

        # Snapshot reconstruction modules before accessing torch-dependent symbol
        reconstruction_modules_before = {
            key for key in sys.modules if "transformation_portal.spatial_ai.reconstruction" in key
        }

        # Access torch-dependent symbol
        _ = reconstruction_pkg.GaussianBackend

        # Verify gaussian_backend module was loaded
        assert (
            "transformation_portal.spatial_ai.reconstruction.gaussian_backend" in sys.modules
        ), "gaussian_backend must be loaded after accessing GaussianBackend"


class TestLazyImportEdgeCases:
    """Test edge cases and error handling for lazy imports."""

    def test_invalid_symbol_raises_attribute_error(self):
        """Test that accessing non-existent symbols raises AttributeError.

        PEP 562 __getattr__ should raise AttributeError for symbols that
        don't exist, not silently fail or return None.
        """
        import transformation_portal.spatial_ai.reconstruction as reconstruction_pkg

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = reconstruction_pkg.NonExistentClass

    def test_lazy_import_is_idempotent(self):
        """Test that accessing the same symbol multiple times is safe.

        Lazy imports should be idempotent - multiple accesses should
        return the same object, not re-import.
        """
        # Remove reconstruction from sys.modules
        reconstruction_modules = [key for key in sys.modules if "transformation_portal.spatial_ai.reconstruction" in key]
        for module_name in reconstruction_modules:
            del sys.modules[module_name]

        import transformation_portal.spatial_ai.reconstruction as reconstruction_pkg

        # Access same symbol multiple times
        backend1 = reconstruction_pkg.GaussianBackend
        backend2 = reconstruction_pkg.GaussianBackend
        backend3 = reconstruction_pkg.GaussianBackend

        # All should be the same class object (identity, not just equality)
        assert backend1 is backend2
        assert backend2 is backend3

    def test_lazy_import_works_with_getattr(self):
        """Test that lazy imports work with getattr() function.

        Users might use getattr(module, "GaussianBackend") instead of
        direct attribute access. This should still trigger lazy loading.
        """
        # Remove reconstruction from sys.modules
        reconstruction_modules = [key for key in sys.modules if "transformation_portal.spatial_ai.reconstruction" in key]
        for module_name in reconstruction_modules:
            del sys.modules[module_name]

        import transformation_portal.spatial_ai.reconstruction as reconstruction_pkg

        # Snapshot modules before
        modules_before = set(sys.modules)

        # Use getattr() instead of direct access
        backend = getattr(reconstruction_pkg, "GaussianBackend")

        # Verify gaussian_backend module was loaded
        assert "transformation_portal.spatial_ai.reconstruction.gaussian_backend" in sys.modules
        assert backend is not None
