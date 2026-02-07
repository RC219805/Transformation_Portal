"""Zone resolution for APEX workflow multi-zone testing.

Zones represent fault/latency/capacity boundaries in the deployment topology:
- Kubernetes topology zones (node.kubernetes.io/zone)
- AWS availability zones (EC2 instance metadata)
- On-prem racks/data centers
- Local development (fallback to "local")

Design:
- Priority-based resolution with clear fallback chain
- Environment variable override for explicit control
- Caching to avoid repeated metadata queries
- Safe fallbacks (never raises, always returns a string)

Usage:
    from transformation_portal.metrics.zone_resolver import ZoneResolver

    # Automatic resolution
    zone = ZoneResolver.resolve()

    # Explicit override (for testing)
    zone = ZoneResolver.resolve(override="test-zone-1")

    # Check if running in multi-zone environment
    is_multi_zone = ZoneResolver.is_multi_zone()

Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
from typing import Optional

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


class ZoneResolver:
    """Resolver for deployment zone detection.

    Resolution priority:
    1. Explicit override parameter
    2. Environment variable APEX_ZONE
    3. Kubernetes topology zone (KUBE_NODE_ZONE or downward API)
    4. AWS EC2 availability zone (instance metadata)
    5. "local" fallback
    """

    _cached_zone: Optional[str] = None

    @classmethod
    def resolve(cls, override: Optional[str] = None) -> str:
        """Resolve the current zone.

        Args:
            override: Explicit zone override (for testing)

        Returns:
            Zone identifier (never None, always returns a string)
        """
        # 1. Explicit override (highest priority)
        if override:
            logger.debug(f"Zone resolved via override: {override}")
            return override

        # 2. Check cache (for efficiency)
        if cls._cached_zone:
            return cls._cached_zone

        # 3. Environment variable APEX_ZONE
        apex_zone = os.environ.get("APEX_ZONE")
        if apex_zone:
            cls._cached_zone = apex_zone
            logger.info(f"Zone resolved via APEX_ZONE: {apex_zone}")
            return apex_zone

        # 4. Kubernetes topology zone
        kube_zone = cls._resolve_kubernetes_zone()
        if kube_zone:
            cls._cached_zone = kube_zone
            logger.info(f"Zone resolved via Kubernetes: {kube_zone}")
            return kube_zone

        # 5. AWS EC2 availability zone
        aws_zone = cls._resolve_aws_zone()
        if aws_zone:
            cls._cached_zone = aws_zone
            logger.info(f"Zone resolved via AWS metadata: {aws_zone}")
            return aws_zone

        # 6. Fallback to "local"
        cls._cached_zone = "local"
        logger.info("Zone resolved to fallback: local")
        return "local"

    @classmethod
    def _resolve_kubernetes_zone(cls) -> Optional[str]:
        """Resolve Kubernetes topology zone.

        Checks:
        1. KUBE_NODE_ZONE (common convention)
        2. Downward API mounted at /etc/podinfo/zone

        Returns:
            Zone identifier or None
        """
        # Check environment variable (common Downward API pattern)
        kube_zone = os.environ.get("KUBE_NODE_ZONE")
        if kube_zone:
            return kube_zone

        # Check downward API mount
        try:
            pod_zone_file = "/etc/podinfo/zone"
            if os.path.exists(pod_zone_file):
                with open(pod_zone_file, "r") as f:
                    zone = f.read().strip()
                    if zone:
                        return zone
        except Exception as e:
            logger.debug(f"Failed to read Kubernetes zone from {pod_zone_file}: {e}")

        return None

    @classmethod
    def _resolve_aws_zone(cls) -> Optional[str]:
        """Resolve AWS EC2 availability zone from instance metadata.

        Uses IMDSv2 (token-based) for security.

        Returns:
            Availability zone (e.g., "us-west-2a") or None
        """
        try:
            import urllib.error
            import urllib.request

            # IMDSv2: Get token first
            token_url = "http://169.254.169.254/latest/api/token"
            token_req = urllib.request.Request(
                token_url, headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, method="PUT"
            )

            try:
                with urllib.request.urlopen(token_req, timeout=1) as response:
                    token = response.read().decode()
            except (urllib.error.URLError, TimeoutError):
                # Not running on EC2 or IMDSv2 not available
                return None

            # Get availability zone using token
            az_url = "http://169.254.169.254/latest/meta-data/placement/availability-zone"
            az_req = urllib.request.Request(az_url, headers={"X-aws-ec2-metadata-token": token})

            with urllib.request.urlopen(az_req, timeout=1) as response:
                az = response.read().decode().strip()
                if az:
                    return az

        except Exception as e:
            logger.debug(f"Failed to resolve AWS zone: {e}")

        return None

    @classmethod
    def is_multi_zone(cls) -> bool:
        """Check if running in a multi-zone environment.

        Returns:
            True if zone is not "local"
        """
        zone = cls.resolve()
        return zone != "local"

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached zone (for testing)."""
        cls._cached_zone = None
