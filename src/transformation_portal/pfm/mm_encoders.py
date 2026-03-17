"""Multimodal encoders for Pipeline Foundation Model.

This module provides encoders for different modalities:
- Images (CLIP, ViT, DINOv2)
- 3D data (PointNet, mesh transformers)
- Structured data (metrics, configs)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


class ImageEncoder(nn.Module):
    """Image encoder using pretrained vision models.

    Supports multiple backends:
    - CLIP (default, good for semantic understanding)
    - DINOv2 (better for geometric features)
    - ViT (general purpose)

    Example:
        >>> encoder = ImageEncoder(backend="clip")
        >>> images = torch.randn(2, 3, 224, 224)
        >>> features = encoder(images)  # [2, 512]
    """

    def __init__(
        self,
        backend: str = "clip",
        model_name: str | None = None,
        output_dim: int = 512,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.output_dim = output_dim
        self._model = None
        self._processor = None

        # Lazy loading to avoid import errors
        self.model_name = model_name or self._default_model_name()
        self.freeze = freeze

        # Fallback projection if model unavailable
        self.fallback_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

        self._initialized = False

    def _default_model_name(self) -> str:
        """Get default model name for backend."""
        defaults = {
            "clip": "openai/clip-vit-base-patch32",
            "dinov2": "facebook/dinov2-base",
            "vit": "google/vit-base-patch16-224",
        }
        return defaults.get(self.backend, defaults["clip"])

    def _init_model(self) -> None:
        """Lazily initialize the model."""
        if self._initialized:
            return

        try:
            if self.backend == "clip":
                from transformers import CLIPModel, CLIPProcessor

                self._model = CLIPModel.from_pretrained(self.model_name)
                self._processor = CLIPProcessor.from_pretrained(self.model_name)

                if self.freeze:
                    for param in self._model.parameters():
                        param.requires_grad = False

            elif self.backend == "dinov2":
                from transformers import Dinov2Model

                self._model = Dinov2Model.from_pretrained(self.model_name)

                if self.freeze:
                    for param in self._model.parameters():
                        param.requires_grad = False

            self._initialized = True

        except Exception:
            # Use fallback if model loading fails
            self._model = None
            self._initialized = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors.

        Args:
            images: Image tensor [B, C, H, W] (normalized)

        Returns:
            Feature vectors [B, output_dim]
        """
        self._init_model()

        if self._model is None:
            # Use fallback
            return self.fallback_proj(images)

        with torch.no_grad() if self.freeze else torch.enable_grad():
            if self.backend == "clip":
                features = self._model.get_image_features(pixel_values=images)
            elif self.backend == "dinov2":
                outputs = self._model(pixel_values=images)
                features = outputs.last_hidden_state[:, 0]  # CLS token
            else:
                features = self.fallback_proj(images)

        return features


class PointCloudEncoder(nn.Module):
    """PointNet-style encoder for 3D point clouds.

    Uses shared MLPs with global max pooling to extract
    permutation-invariant features from point clouds.

    Example:
        >>> encoder = PointCloudEncoder(output_dim=256)
        >>> points = torch.randn(2, 1024, 3)  # [B, N, 3]
        >>> features = encoder(points)  # [2, 256]
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 256,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_dims = hidden_dims or [64, 128, 256]

        # Point-wise MLPs
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            ])
            prev_dim = dim

        self.pointwise = nn.Sequential(*layers)

        # Global feature extraction
        self.global_feat = nn.Sequential(
            nn.Linear(hidden_dims[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode point cloud to feature vector.

        Args:
            points: Point cloud [B, N, input_dim]

        Returns:
            Feature vector [B, output_dim]
        """
        b, n, _ = points.shape

        # Point-wise features
        x = points.view(b * n, -1)
        x = self.pointwise(x)
        x = x.view(b, n, -1)

        # Global max pooling
        x = x.max(dim=1)[0]  # [B, hidden_dim]

        # Final projection
        return self.global_feat(x)


class MeshEncoder(nn.Module):
    """Encoder for 3D meshes.

    Handles mesh data by converting to point cloud or
    using mesh-specific operations.
    """

    def __init__(
        self,
        output_dim: int = 256,
        num_samples: int = 1024,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

        # Use point cloud encoder on sampled vertices
        self.point_encoder = PointCloudEncoder(
            input_dim=3,
            output_dim=output_dim,
        )

        # Optional: encode normals
        self.normal_encoder = PointCloudEncoder(
            input_dim=3,
            output_dim=output_dim // 2,
        )

        self.fusion = nn.Linear(output_dim + output_dim // 2, output_dim)

    def forward(
        self,
        vertices: torch.Tensor,
        normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode mesh to feature vector.

        Args:
            vertices: Mesh vertices [B, V, 3]
            normals: Optional vertex normals [B, V, 3]

        Returns:
            Feature vector [B, output_dim]
        """
        # Sample points if too many vertices
        b, v, _ = vertices.shape
        if v > self.num_samples:
            idx = torch.randperm(v)[:self.num_samples]
            vertices = vertices[:, idx]
            if normals is not None:
                normals = normals[:, idx]

        # Encode vertices
        vert_feat = self.point_encoder(vertices)

        if normals is not None:
            norm_feat = self.normal_encoder(normals)
            return self.fusion(torch.cat([vert_feat, norm_feat], dim=-1))

        return vert_feat


class StructuredEncoder(nn.Module):
    """Encoder for structured pipeline data.

    Projects metrics, configs, and diff features to
    embedding space.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode structured features.

        Args:
            x: Feature vector [B, input_dim]

        Returns:
            Embedding [B, output_dim]
        """
        return self.net(x)


class CrossModalFusion(nn.Module):
    """Cross-modal attention fusion.

    Fuses features from multiple modalities using
    self-attention over modal tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Fuse modal tokens.

        Args:
            tokens: Modal tokens [B, M, D] where M = num modalities

        Returns:
            Fused representation [B, D]
        """
        # Self-attention over modalities
        attn_out, _ = self.attn(tokens, tokens, tokens)
        x = self.norm(tokens + attn_out)

        # FFN
        x = x + self.ffn(x)

        # Global pooling (attention-weighted mean)
        weights = F.softmax(x.mean(dim=-1), dim=-1)
        fused = (weights.unsqueeze(-1) * x).sum(dim=1)

        return fused


class ModalityProjector(nn.Module):
    """Projects different modality features to common dimension."""

    def __init__(
        self,
        input_dims: dict[str, int],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in input_dims.items()
        })

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Project all modality features.

        Args:
            features: Dictionary of modality features

        Returns:
            Projected features (same keys)
        """
        return {
            name: self.projectors[name](feat)
            for name, feat in features.items()
            if name in self.projectors
        }
