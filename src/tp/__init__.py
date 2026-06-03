"""Public contract, fixity, and Phase 4 tooling import surface.

The ``tp`` package is intentionally separate from
``transformation_portal``. Keep this surface lightweight, deterministic, and
limited to contract-bearing helpers such as ``tp.crypto``, ``tp.merkle``, and
``tp.phase4``.
"""

__all__ = ["crypto", "merkle", "phase4"]
