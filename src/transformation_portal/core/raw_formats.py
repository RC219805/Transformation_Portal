"""Single source of truth for RAW camera file extensions.

Lives in ``core/`` (rather than under ``lux_depth_v3``) so format-classifying
modules that must avoid heavy imports — notably
``lux_depth_v3.input_manager`` (no PIL/rawpy at import time, enforced by
``test_no_eager_heavy_imports``) — can share this constant with the
rendering loader and the ingest sidecar generator without pulling in their
heavier dependency surface.

Only stdlib imports are permitted here. Adding ``numpy``/``PIL``/``rawpy``
imports would re-introduce the very import-weight regression this module
exists to prevent.
"""

from __future__ import annotations

# RAW file extensions (case-insensitive). When extending, mirror the change
# in ``docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md`` and the loader docstring; tests
# guard parity across the modules that re-export this constant
# (``lux_depth_v3.raw_loader``, ``ingest.raw_sidecar``,
# ``lux_depth_v3.input_manager``).
RAW_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Canon
        ".cr2",
        ".cr3",
        ".crw",
        # Nikon
        ".nef",
        ".nrw",
        # Sony
        ".arw",
        ".srf",
        ".sr2",
        ".srw",
        # Adobe
        ".dng",
        # Olympus
        ".orf",
        # Fujifilm
        ".raf",
        # Pentax
        ".pef",
        # Panasonic
        ".rw2",
        # Phase One
        ".iiq",
        # Hasselblad
        ".3fr",
        # Note: DNG is TIFF-based RAW format (included above).
        # Standard TIFF (.tif/.tiff) is NOT RAW and handled via PIL.
    }
)
