"""Preprocessing module for DA3 pipeline.

Handles image resizing, normalization, and preparation for inference.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, Union

import numpy as np
import torch
from PIL import Image

from lux_depth_v3.config import PreprocessingConfig


class Preprocessor:
    """Image preprocessor for DA3 inference."""
    
    # ImageNet normalization constants
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
    
    def preprocess(
        self,
        image: np.ndarray,
        return_tensors: bool = True,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
        """Preprocess image for inference.
        
        Args:
            image: Input image (H, W, C) in range [0, 255] uint8
            return_tensors: Return PyTorch tensor if True, else numpy array
        
        Returns:
            Tuple of (preprocessed_image, metadata)
            - preprocessed_image: (C, H, W) tensor or array
            - metadata: dict with original_size, target_size, etc.
        """
        original_size = image.shape[:2]  # (H, W)
        
        # Convert to float32 [0, 1]
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)
        
        # Resize if needed
        if self.config.target_size is not None:
            image_float = self._resize(image_float, self.config.target_size)
            target_size = self.config.target_size
        else:
            target_size = original_size
        
        # Pad to multiple if needed
        if self.config.pad_to_multiple > 1:
            image_float, padding = self._pad_to_multiple(
                image_float,
                self.config.pad_to_multiple
            )
        else:
            padding = (0, 0, 0, 0)
        
        # Normalize
        if self.config.normalize:
            image_float = self._normalize(image_float)
        
        # Convert to tensor (C, H, W)
        if image_float.ndim == 2:
            image_float = image_float[..., np.newaxis]
        
        image_chw = np.transpose(image_float, (2, 0, 1))
        
        if return_tensors:
            image_tensor = torch.from_numpy(image_chw)
            result = image_tensor
        else:
            result = image_chw
        
        metadata = {
            "original_size": original_size,
            "target_size": target_size,
            "padding": padding,
            "normalized": self.config.normalize,
        }
        
        return result, metadata
    
    def _resize(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Resize image to target size.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            target_size: Target size (height, width)
        
        Returns:
            Resized image
        """
        h_target, w_target = target_size
        h_orig, w_orig = image.shape[:2]
        
        if (h_orig, w_orig) == (h_target, w_target):
            return image
        
        # Determine resize mode
        resize_modes = {
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        resample = resize_modes.get(self.config.resize_mode, Image.BILINEAR)
        
        # Handle aspect ratio
        if self.config.maintain_aspect:
            # Compute scale to fit within target size
            scale = min(w_target / w_orig, h_target / h_orig)
            w_new = int(w_orig * scale)
            h_new = int(h_orig * scale)
        else:
            w_new = w_target
            h_new = h_target
        
        # Convert to PIL, resize, convert back
        if image.ndim == 2:
            pil_img = Image.fromarray((image * 255).astype(np.uint8), mode="L")
        else:
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
        
        pil_resized = pil_img.resize((w_new, h_new), resample=resample)
        resized = np.array(pil_resized).astype(np.float32) / 255.0
        
        # Center pad if maintaining aspect ratio
        if self.config.maintain_aspect and (w_new != w_target or h_new != h_target):
            padded = np.zeros((h_target, w_target, image.shape[2] if image.ndim == 3 else 1), dtype=np.float32)
            y_offset = (h_target - h_new) // 2
            x_offset = (w_target - w_new) // 2
            
            if resized.ndim == 2:
                padded[y_offset:y_offset+h_new, x_offset:x_offset+w_new, 0] = resized
            else:
                padded[y_offset:y_offset+h_new, x_offset:x_offset+w_new] = resized
            
            resized = padded
        
        return resized
    
    def _pad_to_multiple(
        self,
        image: np.ndarray,
        multiple: int,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Pad image to be multiple of N.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            multiple: Multiple to pad to
        
        Returns:
            Tuple of (padded_image, padding)
            - padding: (top, bottom, left, right)
        """
        h, w = image.shape[:2]
        
        # Compute padding
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        if pad_h == 0 and pad_w == 0:
            return image, (0, 0, 0, 0)
        
        # Center padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad
        if image.ndim == 2:
            padded = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="edge",
            )
        else:
            padded = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )
        
        return padded, (pad_top, pad_bottom, pad_left, pad_right)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply ImageNet normalization.
        
        Args:
            image: Input image (H, W, C) in range [0, 1]
        
        Returns:
            Normalized image
        """
        if image.ndim == 2:
            # Grayscale - use mean of RGB normalization
            mean = self.IMAGENET_MEAN.mean()
            std = self.IMAGENET_STD.mean()
            return (image - mean) / std
        
        # RGB normalization
        normalized = (image - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return normalized
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """Reverse ImageNet normalization.
        
        Args:
            image: Normalized image (H, W, C)
        
        Returns:
            Denormalized image in range [0, 1]
        """
        if image.ndim == 2:
            mean = self.IMAGENET_MEAN.mean()
            std = self.IMAGENET_STD.mean()
            return image * std + mean
        
        denormalized = image * self.IMAGENET_STD + self.IMAGENET_MEAN
        return np.clip(denormalized, 0, 1)
    
    def unpad(
        self,
        image: np.ndarray,
        padding: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Remove padding from image.
        
        Args:
            image: Padded image
            padding: Padding (top, bottom, left, right)
        
        Returns:
            Unpadded image
        """
        top, bottom, left, right = padding
        
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            return image
        
        h, w = image.shape[:2]
        h_unpad = h - top - bottom
        w_unpad = w - left - right
        
        if image.ndim == 2:
            return image[top:top+h_unpad, left:left+w_unpad]
        else:
            return image[top:top+h_unpad, left:left+w_unpad, :]
