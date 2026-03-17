"""PFM Tokenizer for structured data to tensor conversion.

Converts pipeline step records into tensor representations
suitable for transformer processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TokenizerConfig:
    """Configuration for PFM tokenizer."""

    # Vocabulary sizes
    max_nodes: int = 32
    max_actions: int = 64

    # Feature dimensions
    metric_dim: int = 5
    config_dim: int = 6
    diff_dim: int = 5

    # Special tokens
    pad_token: int = 0
    unk_token: int = 1
    start_token: int = 2
    end_token: int = 3


class PFMTokenizer:
    """Tokenizer for Pipeline Foundation Model.

    Converts structured step dictionaries into tensor representations.
    Handles:
    - Node IDs → embedding indices
    - Metrics → normalized float vectors
    - Configs → normalized float vectors
    - Diffs → count vectors
    - Artifacts → hash embeddings (optional)

    Example:
        >>> tokenizer = PFMTokenizer({"sam2": 4, "nvdiffrec": 5})
        >>> step = {"node_id": "sam2", "metrics": {"score": 0.8}}
        >>> node_id, features = tokenizer.encode_step(step)
    """

    def __init__(
        self,
        node_vocab: dict[str, int] | None = None,
        config: TokenizerConfig | None = None,
    ) -> None:
        self.config = config or TokenizerConfig()
        self.node_vocab = node_vocab or {}

        # Default node vocabulary if not provided
        if not self.node_vocab:
            self.node_vocab = {
                "<pad>": 0,
                "<unk>": 1,
                "sam2": 2,
                "nvdiffrec": 3,
                "material_backend": 4,
                "depth_backend": 5,
                "llava": 6,
                "apex_eval": 7,
            }

        # Metric keys in order
        self.metric_keys = ["score", "psnr", "lpips", "llava", "ssim"]

        # Config keys in order
        self.config_keys = [
            "threshold",
            "steps",
            "iterations",
            "roughness_bias",
            "metalness",
            "exposure",
        ]

        # Diff types in order
        self.diff_types = ["geometry", "texture", "missing", "artifact", "semantic"]

    def encode_step(self, step: dict[str, Any]) -> tuple[int, torch.Tensor]:
        """Encode a single step to tensors.

        Args:
            step: Step dictionary with node_id, metrics, config, diff

        Returns:
            Tuple of (node_id_index, feature_vector)
        """
        # Encode node ID
        node_id = step.get("node_id", "<unk>")
        node_idx = self.node_vocab.get(node_id, self.config.unk_token)

        # Encode metrics
        metrics = step.get("metrics", {})
        metric_vec = [metrics.get(k, 0.0) for k in self.metric_keys]

        # Encode config
        cfg = step.get("config", {})
        config_vec = [cfg.get(k, 0.0) for k in self.config_keys]

        # Encode diff counts
        diff = step.get("diff", {})
        diff_vec = [diff.get(k, 0) for k in self.diff_types]

        # Concatenate all features
        features = metric_vec + config_vec + diff_vec

        return node_idx, torch.tensor(features, dtype=torch.float32)

    def encode_sequence(self, sequence: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a sequence of steps.

        Args:
            sequence: List of step dictionaries

        Returns:
            Tuple of (node_ids [T], features [T, F])
        """
        node_ids = []
        features = []

        for step in sequence:
            nid, feat = self.encode_step(step)
            node_ids.append(nid)
            features.append(feat)

        return (
            torch.tensor(node_ids, dtype=torch.long),
            torch.stack(features),
        )

    def encode_batch(
        self,
        batch: list[list[dict[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of sequences.

        Args:
            batch: List of sequences

        Returns:
            Tuple of (node_ids [B, T], features [B, T, F])
        """
        all_node_ids = []
        all_features = []

        for seq in batch:
            nids, feats = self.encode_sequence(seq)
            all_node_ids.append(nids)
            all_features.append(feats)

        return (
            torch.stack(all_node_ids),
            torch.stack(all_features),
        )

    def get_feature_dim(self) -> int:
        """Return total feature dimension."""
        return self.config.metric_dim + self.config.config_dim + self.config.diff_dim

    def add_node(self, node_id: str) -> int:
        """Add new node to vocabulary.

        Args:
            node_id: Node identifier

        Returns:
            Assigned index
        """
        if node_id not in self.node_vocab:
            idx = len(self.node_vocab)
            self.node_vocab[node_id] = idx
        return self.node_vocab[node_id]

    def decode_node(self, idx: int) -> str:
        """Decode node index to ID.

        Args:
            idx: Node index

        Returns:
            Node ID string
        """
        reverse_vocab = {v: k for k, v in self.node_vocab.items()}
        return reverse_vocab.get(idx, "<unk>")


class MultimodalTokenizer(PFMTokenizer):
    """Extended tokenizer with multimodal support.

    Handles image and 3D mesh embeddings in addition to
    structured pipeline data.
    """

    def __init__(
        self,
        node_vocab: dict[str, int] | None = None,
        config: TokenizerConfig | None = None,
        image_embed_dim: int = 512,
        mesh_embed_dim: int = 256,
    ) -> None:
        super().__init__(node_vocab, config)
        self.image_embed_dim = image_embed_dim
        self.mesh_embed_dim = mesh_embed_dim

        # Placeholder for external encoders
        self._image_encoder = None
        self._mesh_encoder = None

    def set_image_encoder(self, encoder: Any) -> None:
        """Set image encoder (e.g., CLIP, DINOv2)."""
        self._image_encoder = encoder

    def set_mesh_encoder(self, encoder: Any) -> None:
        """Set mesh encoder (e.g., PointNet, MeshEncoder)."""
        self._mesh_encoder = encoder

    def encode_step_multimodal(
        self,
        step: dict[str, Any],
        image: torch.Tensor | None = None,
        mesh: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode step with optional multimodal inputs.

        Args:
            step: Step dictionary
            image: Optional image tensor [C, H, W]
            mesh: Optional mesh features [V, F] or embedding

        Returns:
            Tuple of (node_idx, features, image_embed, mesh_embed)
        """
        node_idx, features = self.encode_step(step)

        image_embed = None
        mesh_embed = None

        if image is not None and self._image_encoder is not None:
            with torch.no_grad():
                image_embed = self._image_encoder(image.unsqueeze(0)).squeeze(0)

        if mesh is not None and self._mesh_encoder is not None:
            with torch.no_grad():
                mesh_embed = self._mesh_encoder(mesh.unsqueeze(0)).squeeze(0)

        return node_idx, features, image_embed, mesh_embed
