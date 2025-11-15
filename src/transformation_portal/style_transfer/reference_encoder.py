"""Reference image encoder for style feature extraction.

Provides utilities for encoding reference images into style features
that can be used for IP-Adapter style transfer. Supports:
- Single image encoding
- Multi-image averaging
- Style feature caching
- Batch encoding
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle

import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False


logger = logging.getLogger(__name__)


class ReferenceImageEncoder:
    """Encoder for extracting style features from reference images.

    Uses CLIP vision model to encode images into feature vectors
    that capture visual style characteristics.

    Example:
        >>> encoder = ReferenceImageEncoder()
        >>>
        >>> # Encode single image
        >>> features = encoder.encode("reference.jpg")
        >>>
        >>> # Encode collection and average
        >>> features = encoder.encode_collection([
        ...     "ref1.jpg", "ref2.jpg", "ref3.jpg"
        ... ])
        >>>
        >>> # Save for reuse
        >>> encoder.save_features(features, "my_style.pkl")
    """

    MODEL_NAME = "openai/clip-vit-large-patch14"

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
        cache_dir: Optional[Path] = None
    ):
        """Initialize reference image encoder.

        Args:
            device: Computation device
            torch_dtype: Tensor dtype
            cache_dir: Directory for caching features
        """
        if not ENCODER_AVAILABLE:
            raise ImportError(
                "Reference encoder requires transformers. "
                "Install with: pip install transformers>=4.38.0"
            )

        self.device = device or self._detect_device()
        self.torch_dtype = torch_dtype
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing reference encoder on {self.device}")

        # Load CLIP vision model
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch_dtype
        ).to(self.device)

        self.processor = CLIPImageProcessor.from_pretrained(self.MODEL_NAME)

        logger.info("Reference encoder initialized")

    def _detect_device(self) -> str:
        """Auto-detect optimal device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        normalize: bool = True
    ) -> torch.Tensor:
        """Encode image to style features.

        Args:
            image: Input image
            normalize: Normalize features to unit length

        Returns:
            Style feature tensor
        """
        # Load image
        pil_image = self._load_image(image)

        # Preprocess
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)

        # Encode
        with torch.inference_mode():
            features = self.model(**inputs).image_embeds

        # Normalize if requested
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)

        return features

    def encode_collection(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        weights: Optional[List[float]] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """Encode collection of images and average.

        Args:
            images: List of images
            weights: Optional weights for averaging
            normalize: Normalize final features

        Returns:
            Averaged style features
        """
        logger.info(f"Encoding collection of {len(images)} images")

        # Encode all images
        features_list = [self.encode(img, normalize=False) for img in images]

        # Set equal weights if not provided
        if weights is None:
            weights = [1.0 / len(images)] * len(images)

        # Normalize weights
        weights = torch.tensor(weights, device=self.device)
        weights = weights / weights.sum()

        # Weighted average
        averaged = sum(
            features * weight
            for features, weight in zip(features_list, weights)
        )

        # Normalize if requested
        if normalize:
            averaged = averaged / averaged.norm(dim=-1, keepdim=True)

        logger.info("Collection encoded and averaged")

        return averaged

    def encode_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 8
    ) -> torch.Tensor:
        """Encode batch of images efficiently.

        Args:
            images: List of images
            batch_size: Batch size for processing

        Returns:
            Stacked feature tensors
        """
        logger.info(f"Batch encoding {len(images)} images")

        all_features = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Load batch
            pil_images = [self._load_image(img) for img in batch]

            # Preprocess batch
            inputs = self.processor(
                images=pil_images,
                return_tensors="pt"
            ).to(self.device)

            # Encode batch
            with torch.inference_mode():
                features = self.model(**inputs).image_embeds

            all_features.append(features)

        # Stack all features
        stacked = torch.cat(all_features, dim=0)

        logger.info(f"Batch encoding complete: {stacked.shape}")

        return stacked

    def save_features(
        self,
        features: torch.Tensor,
        path: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> None:
        """Save encoded features to file.

        Args:
            features: Feature tensor to save
            path: Output file path
            metadata: Optional metadata to save with features
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "features": features.cpu().numpy(),
            "shape": features.shape,
            "dtype": str(features.dtype),
            "metadata": metadata or {}
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Features saved to {path}")

    def load_features(
        self,
        path: Union[str, Path]
    ) -> Tuple[torch.Tensor, Dict]:
        """Load encoded features from file.

        Args:
            path: Input file path

        Returns:
            Tuple of (features tensor, metadata dict)
        """
        path = Path(path)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        features = torch.from_numpy(data["features"]).to(self.device)
        metadata = data.get("metadata", {})

        logger.info(f"Features loaded from {path}")

        return features, metadata

    def compute_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> float:
        """Compute cosine similarity between feature vectors.

        Args:
            features1: First feature tensor
            features2: Second feature tensor

        Returns:
            Similarity score (0-1)
        """
        similarity = torch.nn.functional.cosine_similarity(
            features1,
            features2,
            dim=-1
        ).item()

        # Normalize to 0-1
        return (similarity + 1) / 2

    def find_most_similar(
        self,
        query_features: torch.Tensor,
        reference_features: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find most similar references to query.

        Args:
            query_features: Query feature tensor
            reference_features: Reference features (batch)
            top_k: Number of top matches to return

        Returns:
            List of (index, similarity) tuples
        """
        # Compute similarities
        similarities = torch.nn.functional.cosine_similarity(
            query_features.unsqueeze(0),
            reference_features,
            dim=-1
        )

        # Normalize to 0-1
        similarities = (similarities + 1) / 2

        # Get top-k
        top_k = min(top_k, len(similarities))
        values, indices = torch.topk(similarities, top_k)

        results = [
            (idx.item(), val.item())
            for idx, val in zip(indices, values)
        ]

        return results

    def create_style_library(
        self,
        reference_dir: Union[str, Path],
        output_path: Union[str, Path],
        pattern: str = "*.jpg"
    ) -> None:
        """Create style library from directory of references.

        Args:
            reference_dir: Directory containing reference images
            output_path: Output file for style library
            pattern: File pattern for matching images
        """
        reference_dir = Path(reference_dir)
        logger.info(f"Creating style library from {reference_dir}")

        # Find all matching images
        image_paths = list(reference_dir.glob(pattern))

        if not image_paths:
            raise ValueError(f"No images found matching {pattern}")

        logger.info(f"Found {len(image_paths)} images")

        # Encode all images
        features = self.encode_batch(image_paths)

        # Create metadata
        metadata = {
            "num_images": len(image_paths),
            "image_paths": [str(p) for p in image_paths],
            "pattern": pattern,
            "source_dir": str(reference_dir)
        }

        # Save library
        self.save_features(features, output_path, metadata)

        logger.info(f"Style library created with {len(image_paths)} references")

    def _load_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Load image as PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def __repr__(self) -> str:
        return f"ReferenceImageEncoder(device='{self.device}')"


# Export
__all__ = ['ReferenceImageEncoder']
