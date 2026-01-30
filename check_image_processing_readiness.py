"""
Backward-compatible shim.

Historically, readiness checks lived as a top-level module.
Canonical location is now:
    transformation_portal.readiness.check_image_processing_readiness
"""
from transformation_portal.readiness.check_image_processing_readiness import *  # noqa: F403,F401
