"""Tests for runtime protocol validation in Phase 5F."""

from typing import Optional

import numpy as np
import pytest

from transformation_portal.spatial_ai.materials.contracts import MaterialGenerationConfig, PBRTextures
from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend
from transformation_portal.spatial_ai.materials.protocol import validate_backend_protocol



pytestmark = pytest.mark.unit

class MissingMethodBackend:
    """Backend missing required protocol method."""


class MissingDefaultsBackend:
    """Backend with required names but invalid optional parameter defaults."""

    def generate_pbr_textures(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        material_hint: Optional[str],
        config: Optional[MaterialGenerationConfig],
    ) -> PBRTextures:
        raise NotImplementedError


class WrongReturnTypeBackend:
    """Backend with wrong return type annotation."""

    def generate_pbr_textures(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        material_hint: Optional[str] = None,
        config: Optional[MaterialGenerationConfig] = None,
    ) -> np.ndarray:
        return rgb


class Pep604OptionalBackend:
    """Backend using PEP 604 optional unions in annotations."""

    def generate_pbr_textures(
        self,
        rgb: np.ndarray,
        mask: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        material_hint: str | None = None,
        config: MaterialGenerationConfig | None = None,
    ) -> PBRTextures:
        raise NotImplementedError


def test_validate_backend_protocol_accepts_material_backend():
    backend = MaterialBackend(backend="heuristic", device="cpu")
    assert validate_backend_protocol(backend) is True


def test_validate_backend_protocol_rejects_missing_method():
    with pytest.raises(TypeError, match="generate_pbr_textures"):
        validate_backend_protocol(MissingMethodBackend())  # type: ignore[arg-type]


def test_validate_backend_protocol_rejects_missing_optional_defaults():
    with pytest.raises(TypeError, match="must be optional"):
        validate_backend_protocol(MissingDefaultsBackend())  # type: ignore[arg-type]


def test_validate_backend_protocol_rejects_wrong_return_annotation():
    with pytest.raises(TypeError, match="wrong type for 'return'"):
        validate_backend_protocol(WrongReturnTypeBackend())  # type: ignore[arg-type]


def test_validate_backend_protocol_accepts_pep604_optional_annotations():
    assert validate_backend_protocol(Pep604OptionalBackend()) is True
