"""
Experimental features - NOT production-ready.

⚠️ WARNING: This module contains experimental, unstable features.
   - APIs may change without notice
   - No stability guarantees
   - Community support only (best-effort)
   - DO NOT use in production

Import from this module triggers a warning.
"""

import warnings


class ExperimentalWarning(UserWarning):
    """Warning for experimental feature usage."""

    pass


warnings.warn(
    "\n"
    "═══════════════════════════════════════════════════════════════\n"
    "⚠️  EXPERIMENTAL FEATURE WARNING\n"
    "═══════════════════════════════════════════════════════════════\n"
    "\n"
    "You are importing from the 'experimental' module.\n"
    "\n"
    "This code is:\n"
    "  ❌ NOT production-ready\n"
    "  ❌ NOT stable (APIs may change)\n"
    "  ❌ NOT supported (community best-effort)\n"
    "  ❌ NOT recommended for production use\n"
    "\n"
    "Use at your own risk.\n"
    "\n"
    "For production use, see: https://github.com/RC219805/Transformation_Portal/blob/main/QUICKSTART.md\n"
    "═══════════════════════════════════════════════════════════════\n",
    category=ExperimentalWarning,
    stacklevel=2,
)

# Export marker for CI detection
__experimental__ = True
