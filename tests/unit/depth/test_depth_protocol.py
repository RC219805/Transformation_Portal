"""Tests for DepthModel protocol and registry.

These tests validate the depth model protocol interface and the
backend registry system.
"""

from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

from transformation_portal.lux_depth_v3.contracts import DepthArtifact, DepthProvenance, LicenseTier
from transformation_portal.lux_depth_v3.protocols import (
    BackendCapability,
    BackendInfo,
    BackendRole,
    DepthModel,
    DepthModelRegistry,
)


class MockCommercialBackend:
    """Mock commercial depth backend for testing."""

    backend_info = BackendInfo(
        name="Mock Commercial Backend",
        model_id="mock/commercial-v1",
        role=BackendRole.PRODUCTION,
        license_tier=LicenseTier.COMMERCIAL,
        capabilities=frozenset(
            {
                BackendCapability.RELATIVE_DEPTH,
                BackendCapability.BATCH_INFERENCE,
            }
        ),
        description="Test backend for commercial use",
    )

    def __init__(self):
        self._loaded = False

    @property
    def info(self) -> BackendInfo:
        return self.backend_info

    def load(
        self,
        device: str = "auto",
        weights_path: Optional[Path] = None,
        strict_license: bool = True,
    ) -> None:
        self._loaded = True

    def predict(self, image: np.ndarray) -> DepthArtifact:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        h, w = image.shape[:2]
        depth_map = np.random.rand(h, w).astype(np.float32)
        provenance = DepthProvenance(
            model_id=self.info.model_id,
            license_tier=self.info.license_tier,
            device="cpu",
        )
        return DepthArtifact(depth_map=depth_map, provenance=provenance)

    def predict_batch(self, images: List[np.ndarray]) -> List[DepthArtifact]:
        return [self.predict(img) for img in images]


class MockResearchBackend:
    """Mock research backend (non-commercial) for testing."""

    backend_info = BackendInfo(
        name="Mock Research Backend",
        model_id="mock/research-v1",
        role=BackendRole.AUDIT,
        license_tier=LicenseTier.NON_COMMERCIAL,
        capabilities=frozenset(
            {
                BackendCapability.RELATIVE_DEPTH,
                BackendCapability.METRIC_DEPTH,
                BackendCapability.CONFIDENCE_MAP,
            }
        ),
        description="Test backend for research (non-commercial)",
    )

    @property
    def info(self) -> BackendInfo:
        return self.backend_info

    def load(
        self,
        device: str = "auto",
        weights_path: Optional[Path] = None,
        strict_license: bool = True,
    ) -> None:
        pass

    def predict(self, image: np.ndarray) -> DepthArtifact:
        h, w = image.shape[:2]
        depth_map = np.random.rand(h, w).astype(np.float32)
        metric_map = depth_map * 10  # Mock metric depth
        confidence = np.ones_like(depth_map) * 0.9
        provenance = DepthProvenance(
            model_id=self.info.model_id,
            license_tier=self.info.license_tier,
        )
        return DepthArtifact(
            depth_map=depth_map,
            provenance=provenance,
            metric_map_m=metric_map,
            confidence=confidence,
        )


class MockVideoBackend:
    """Mock video-capable backend for testing."""

    backend_info = BackendInfo(
        name="Mock Video Backend",
        model_id="mock/video-v1",
        role=BackendRole.VIDEO,
        license_tier=LicenseTier.COMMERCIAL,
        capabilities=frozenset(
            {
                BackendCapability.RELATIVE_DEPTH,
                BackendCapability.VIDEO_STREAMING,
            }
        ),
        description="Test backend for video processing",
    )

    @property
    def info(self) -> BackendInfo:
        return self.backend_info

    def load(
        self,
        device: str = "auto",
        weights_path: Optional[Path] = None,
        strict_license: bool = True,
    ) -> None:
        pass

    def predict(self, image: np.ndarray) -> DepthArtifact:
        h, w = image.shape[:2]
        depth_map = np.random.rand(h, w).astype(np.float32)
        provenance = DepthProvenance(
            model_id=self.info.model_id,
            license_tier=self.info.license_tier,
        )
        return DepthArtifact(depth_map=depth_map, provenance=provenance)

    def stream_video(
        self,
        frames: Iterator[np.ndarray],
    ) -> Iterator[DepthArtifact]:
        for frame in frames:
            yield self.predict(frame)


class TestBackendRole:
    """Test BackendRole enum."""

    def test_role_values_exist(self):
        """Test all expected roles exist."""
        assert BackendRole.DRAFT
        assert BackendRole.PRODUCTION
        assert BackendRole.VIDEO
        assert BackendRole.AUDIT


class TestBackendCapability:
    """Test BackendCapability enum."""

    def test_capability_values_exist(self):
        """Test all expected capabilities exist."""
        assert BackendCapability.RELATIVE_DEPTH
        assert BackendCapability.METRIC_DEPTH
        assert BackendCapability.CONFIDENCE_MAP
        assert BackendCapability.VIDEO_STREAMING
        assert BackendCapability.BATCH_INFERENCE


class TestBackendInfo:
    """Test BackendInfo dataclass."""

    def test_create_backend_info(self):
        """Test creating backend info."""
        info = BackendInfo(
            name="Test Backend",
            model_id="test/model-v1",
            role=BackendRole.PRODUCTION,
            license_tier=LicenseTier.COMMERCIAL,
        )
        assert info.name == "Test Backend"
        assert info.model_id == "test/model-v1"
        assert info.role == BackendRole.PRODUCTION

    def test_backend_info_supports(self):
        """Test capability support check."""
        info = BackendInfo(
            name="Test",
            model_id="test",
            role=BackendRole.PRODUCTION,
            license_tier=LicenseTier.COMMERCIAL,
            capabilities=frozenset(
                {
                    BackendCapability.RELATIVE_DEPTH,
                    BackendCapability.BATCH_INFERENCE,
                }
            ),
        )
        assert info.supports(BackendCapability.RELATIVE_DEPTH)
        assert info.supports(BackendCapability.BATCH_INFERENCE)
        assert not info.supports(BackendCapability.METRIC_DEPTH)

    def test_backend_info_commercial_safe(self):
        """Test commercial safety check."""
        commercial = BackendInfo(
            name="Commercial",
            model_id="test",
            role=BackendRole.PRODUCTION,
            license_tier=LicenseTier.COMMERCIAL,
        )
        assert commercial.is_commercial_safe()

        research = BackendInfo(
            name="Research",
            model_id="test",
            role=BackendRole.AUDIT,
            license_tier=LicenseTier.NON_COMMERCIAL,
        )
        assert not research.is_commercial_safe()


class TestDepthModelRegistry:
    """Test DepthModelRegistry."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry for each test."""
        return DepthModelRegistry()

    def test_register_backend(self, registry):
        """Test registering a backend."""
        registry.register(MockCommercialBackend)
        backends = registry.list_backends()
        assert len(backends) == 1
        assert backends[0].model_id == "mock/commercial-v1"

    def test_register_with_custom_name(self, registry):
        """Test registering with custom name."""
        registry.register(MockCommercialBackend, name="custom_name")
        backend = registry.get_backend(name="custom_name")
        assert backend.info.model_id == "mock/commercial-v1"

    def test_register_backend_does_not_require_constructor(self, registry):
        """Registration and listing use class metadata without construction."""

        class ConstructorRequiredBackend:
            backend_info = BackendInfo(
                name="Constructor Required",
                model_id="mock/constructor-required",
                role=BackendRole.PRODUCTION,
                license_tier=LicenseTier.COMMERCIAL,
            )

            def __init__(self, required_arg):
                self.required_arg = required_arg

            @property
            def info(self) -> BackendInfo:
                return self.backend_info

            def load(self, device: str = "auto", weights_path: Optional[Path] = None, strict_license: bool = True) -> None:
                pass

            def predict(self, image: np.ndarray) -> DepthArtifact:
                raise NotImplementedError

        registry.register(ConstructorRequiredBackend, name="constructor_required")
        backends = registry.list_backends(role=BackendRole.PRODUCTION)
        assert [info.model_id for info in backends] == ["mock/constructor-required"]

    def test_register_requires_class_available_backend_info(self, registry):
        """Backends must expose metadata without instance construction."""

        class MissingMetadataBackend:
            @property
            def info(self) -> BackendInfo:
                return BackendInfo(
                    name="Missing Metadata Runtime Info",
                    model_id="mock/missing-metadata-runtime-info",
                    role=BackendRole.PRODUCTION,
                    license_tier=LicenseTier.COMMERCIAL,
                )

            def load(self, device: str = "auto", weights_path: Optional[Path] = None, strict_license: bool = True) -> None:
                pass

            def predict(self, image: np.ndarray) -> DepthArtifact:
                raise NotImplementedError

        with pytest.raises(TypeError, match="must expose BackendInfo"):
            registry.register(MissingMetadataBackend)

    def test_register_requires_protocol_info_member(self, registry):
        """Class validation requires the runtime protocol info member."""

        class MissingInfoBackend:
            backend_info = BackendInfo(
                name="Missing Info",
                model_id="mock/missing-info",
                role=BackendRole.PRODUCTION,
                license_tier=LicenseTier.COMMERCIAL,
            )

            def load(self, device: str = "auto", weights_path: Optional[Path] = None, strict_license: bool = True) -> None:
                pass

            def predict(self, image: np.ndarray) -> DepthArtifact:
                raise NotImplementedError

        with pytest.raises(TypeError, match="info"):
            registry.register(MissingInfoBackend)

    def test_list_backends_by_role(self, registry):
        """Test listing backends filtered by role."""
        registry.register(MockCommercialBackend)
        registry.register(MockResearchBackend)
        registry.register(MockVideoBackend)

        production = registry.list_backends(role=BackendRole.PRODUCTION)
        assert len(production) == 1

        video = registry.list_backends(role=BackendRole.VIDEO)
        assert len(video) == 1

        audit = registry.list_backends(role=BackendRole.AUDIT)
        assert len(audit) == 1

    def test_list_backends_commercial_only(self, registry):
        """Test listing only commercial backends."""
        registry.register(MockCommercialBackend)
        registry.register(MockResearchBackend)

        commercial = registry.list_backends(commercial_only=True)
        assert len(commercial) == 1
        assert commercial[0].license_tier == LicenseTier.COMMERCIAL

    def test_get_backend_by_name(self, registry):
        """Test getting backend by name."""
        registry.register(MockCommercialBackend)
        backend = registry.get_backend(name="MockCommercialBackend")
        assert backend.info.name == "Mock Commercial Backend"

    def test_get_backend_by_role(self, registry):
        """Test getting backend by role."""
        registry.register(MockCommercialBackend)
        backend = registry.get_backend(role=BackendRole.PRODUCTION)
        assert backend.info.role == BackendRole.PRODUCTION

    def test_list_and_role_lookup_do_not_instantiate_all_backends(self, registry):
        """Metadata discovery stays constructor-free; acquisition builds only the selected backend."""

        class CountingProductionBackend:
            constructor_calls = 0
            backend_info = BackendInfo(
                name="Counting Production",
                model_id="mock/counting-production",
                role=BackendRole.PRODUCTION,
                license_tier=LicenseTier.COMMERCIAL,
            )

            def __init__(self):
                type(self).constructor_calls += 1

            @property
            def info(self) -> BackendInfo:
                return self.backend_info

            def load(self, device: str = "auto", weights_path: Optional[Path] = None, strict_license: bool = True) -> None:
                pass

            def predict(self, image: np.ndarray) -> DepthArtifact:
                raise NotImplementedError

        class CountingAuditBackend:
            constructor_calls = 0
            backend_info = BackendInfo(
                name="Counting Audit",
                model_id="mock/counting-audit",
                role=BackendRole.AUDIT,
                license_tier=LicenseTier.NON_COMMERCIAL,
            )

            def __init__(self):
                type(self).constructor_calls += 1

            @property
            def info(self) -> BackendInfo:
                return self.backend_info

            def load(self, device: str = "auto", weights_path: Optional[Path] = None, strict_license: bool = True) -> None:
                pass

            def predict(self, image: np.ndarray) -> DepthArtifact:
                raise NotImplementedError

        registry.register(CountingProductionBackend, name="production")
        registry.register(CountingAuditBackend, name="audit")

        listed = registry.list_backends()
        assert len(listed) == 2
        assert CountingProductionBackend.constructor_calls == 0
        assert CountingAuditBackend.constructor_calls == 0

        backend = registry.get_backend(role=BackendRole.PRODUCTION, use_cache=False)
        assert isinstance(backend, CountingProductionBackend)
        assert CountingProductionBackend.constructor_calls == 1
        assert CountingAuditBackend.constructor_calls == 0

    def test_get_backend_commercial_only_rejects_research(self, registry):
        """Test that commercial_only rejects non-commercial backends."""
        registry.register(MockResearchBackend)
        with pytest.raises(ValueError, match="non-commercial license"):
            registry.get_backend(name="MockResearchBackend", commercial_only=True)

    def test_get_backend_commercial_only_rejects_cached_research(self, registry):
        """Commercial filtering is enforced before returning cached instances."""
        registry.register(MockResearchBackend)
        registry.get_backend(name="MockResearchBackend", commercial_only=False)
        with pytest.raises(ValueError, match="non-commercial license"):
            registry.get_backend(name="MockResearchBackend", commercial_only=True)

    def test_get_backend_rechecks_runtime_info_for_commercial_only(self, registry):
        """Registration metadata overrides cannot mask non-commercial runtime info."""
        commercial_override = BackendInfo(
            name="Runtime Override",
            model_id="mock/runtime-override",
            role=BackendRole.PRODUCTION,
            license_tier=LicenseTier.COMMERCIAL,
        )

        class RuntimeResearchBackend:
            constructor_calls = 0
            backend_info = BackendInfo(
                name="Runtime Research",
                model_id="mock/runtime-research",
                role=BackendRole.PRODUCTION,
                license_tier=LicenseTier.NON_COMMERCIAL,
            )

            def __init__(self):
                type(self).constructor_calls += 1

            @property
            def info(self) -> BackendInfo:
                return self.backend_info

            def load(self, device: str = "auto", weights_path: Optional[Path] = None, strict_license: bool = True) -> None:
                pass

            def predict(self, image: np.ndarray) -> DepthArtifact:
                raise NotImplementedError

        registry.register(
            RuntimeResearchBackend,
            name="runtime_research",
            info=commercial_override,
        )
        registry.get_backend(name="runtime_research", commercial_only=False)

        with pytest.raises(ValueError, match="non-commercial license"):
            registry.get_backend(name="runtime_research", commercial_only=True)
        assert RuntimeResearchBackend.constructor_calls == 1

    def test_get_backend_not_found(self, registry):
        """Test error when backend not found."""
        with pytest.raises(KeyError, match="not registered"):
            registry.get_backend(name="NonExistentBackend")

    def test_backend_caching(self, registry):
        """Test that backends are cached."""
        registry.register(MockCommercialBackend)
        backend1 = registry.get_backend(name="MockCommercialBackend")
        backend2 = registry.get_backend(name="MockCommercialBackend")
        assert backend1 is backend2

    def test_backend_no_caching(self, registry):
        """Test that caching can be disabled."""
        registry.register(MockCommercialBackend)
        backend1 = registry.get_backend(
            name="MockCommercialBackend",
            use_cache=False,
        )
        backend2 = registry.get_backend(
            name="MockCommercialBackend",
            use_cache=False,
        )
        assert backend1 is not backend2

    def test_set_fallback_chain(self, registry):
        """Test setting fallback chain."""
        registry.register(MockCommercialBackend)
        registry.register(MockVideoBackend)
        registry.set_fallback_chain(
            BackendRole.PRODUCTION,
            ["MockCommercialBackend", "MockVideoBackend"],
        )
        chain = registry.get_fallback_chain(BackendRole.PRODUCTION)
        assert len(chain) == 2
        assert "MockCommercialBackend" in chain

    def test_set_fallback_chain_invalid_backend(self, registry):
        """Test error when fallback chain has invalid backend."""
        with pytest.raises(KeyError, match="not registered"):
            registry.set_fallback_chain(
                BackendRole.PRODUCTION,
                ["NonExistentBackend"],
            )


class TestMockBackends:
    """Integration tests with mock backends."""

    def test_commercial_backend_predict(self):
        """Test commercial backend prediction."""
        backend = MockCommercialBackend()
        backend.load()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        artifact = backend.predict(image)

        assert artifact.shape == (480, 640)
        assert artifact.is_commercial_safe
        assert artifact.provenance.model_id == "mock/commercial-v1"

    def test_research_backend_metric_depth(self):
        """Test research backend produces metric depth."""
        backend = MockResearchBackend()
        backend.load()
        image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        artifact = backend.predict(image)

        assert artifact.has_metric_depth
        assert artifact.has_confidence
        assert not artifact.is_commercial_safe

    def test_video_backend_streaming(self):
        """Test video backend streaming."""
        backend = MockVideoBackend()
        backend.load()

        def frame_generator():
            for _ in range(5):
                yield np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        artifacts = list(backend.stream_video(frame_generator()))
        assert len(artifacts) == 5
        for artifact in artifacts:
            assert artifact.shape == (100, 100)

    def test_batch_inference(self):
        """Test batch inference capability."""
        backend = MockCommercialBackend()
        backend.load()

        images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        artifacts = backend.predict_batch(images)

        assert len(artifacts) == 3
        for artifact in artifacts:
            assert artifact.shape == (64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
