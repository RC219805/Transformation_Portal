"""IP-Adapter style transfer for architectural photography.

IP-Adapter enables style transfer using reference images, allowing
application of professional architectural photography styles while
preserving structural content.

Key Features:
- Reference-based style transfer
- Architectural photography style presets
- Multi-reference style blending
- Fine-grained style control
- FLUX integration for high-quality results

Use Cases:
- Match client's preferred photography style
- Maintain consistent visual style across property portfolio
- Apply magazine-quality photography aesthetics
- Learn from award-winning architectural photos

Example:
    >>> from transformation_portal.style_transfer import IPAdapterStyleTransfer
    >>>
    >>> # Transfer style from reference photo
    >>> style_transfer = IPAdapterStyleTransfer()
    >>> result = style_transfer.transfer_style(
    ...     content_image="estate.jpg",
    ...     style_reference="architectural_digest_reference.jpg",
    ...     style_strength=0.7
    ... )
    >>>
    >>> # Use preset architectural photography style
    >>> result = style_transfer.apply_preset_style(
    ...     content_image="estate.jpg",
    ...     preset="architectural_digest",
    ...     strength=0.7
    ... )
"""

from transformation_portal.style_transfer.ip_adapter import IPAdapterStyleTransfer
from transformation_portal.style_transfer.style_presets import (
    ArchitecturalStylePresets,
    StylePreset
)
from transformation_portal.style_transfer.reference_encoder import ReferenceImageEncoder
from transformation_portal.style_transfer.multi_reference import MultiReferenceBlender

__all__ = [
    'IPAdapterStyleTransfer',
    'ArchitecturalStylePresets',
    'StylePreset',
    'ReferenceImageEncoder',
    'MultiReferenceBlender',
]
