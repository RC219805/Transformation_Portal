"""Distributed execution client for remote workers.

This module provides client-side functionality for dispatching
tasks to remote workers over TCP.

Example:
    >>> # Send task to worker
    >>> result = send_task("192.168.1.10", 5000, task_dict)
    >>> if result["error"]:
    ...     raise RuntimeError(result["error"])
    >>> outputs = result["outputs"]
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DistributedClientError(RuntimeError):
    """Raised for distributed client errors."""


def send_task(
    host: str,
    port: int,
    task_dict: Dict[str, Any],
    *,
    timeout: float = 300.0,
    buffer_size: int = 1024 * 1024,
) -> Dict[str, Any]:
    """Send a task to a remote worker.

    Args:
        host: Worker hostname or IP
        port: Worker port
        task_dict: Task as dictionary
        timeout: Socket timeout in seconds
        buffer_size: Receive buffer size

    Returns:
        Result dictionary with "outputs" and optional "error"

    Raises:
        DistributedClientError: If communication fails
    """
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            # Send task
            task_json = json.dumps(task_dict).encode("utf-8")
            sock.sendall(task_json)
            sock.shutdown(socket.SHUT_WR)

            # Receive response
            chunks = []
            while True:
                chunk = sock.recv(buffer_size)
                if not chunk:
                    break
                chunks.append(chunk)

            response_data = b"".join(chunks).decode("utf-8")

            if not response_data:
                return {"outputs": {}, "error": "Empty response from worker"}

            return json.loads(response_data)

    except socket.timeout:
        logger.error("Timeout connecting to %s:%d", host, port)
        return {"outputs": {}, "error": f"Timeout connecting to {host}:{port}"}

    except ConnectionRefusedError:
        logger.error("Connection refused: %s:%d", host, port)
        return {"outputs": {}, "error": f"Connection refused: {host}:{port}"}

    except Exception as e:
        logger.error("Failed to send task to %s:%d: %s", host, port, e)
        return {"outputs": {}, "error": str(e)}


class DistributedClient:
    """Client for sending tasks to remote workers.

    Maintains connection state and provides retry logic.

    Example:
        >>> client = DistributedClient("worker-1", 5000)
        >>> result = client.send(task_dict)
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize client.

        Args:
            host: Worker hostname or IP
            port: Worker port
            timeout: Socket timeout
            max_retries: Maximum retry attempts
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self._send_count = 0

    def send(
        self,
        task_dict: Dict[str, Any],
        *,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a task with retry logic.

        Args:
            task_dict: Task as dictionary
            retries: Override max retries

        Returns:
            Result dictionary

        Raises:
            DistributedClientError: If all retries fail
        """
        max_attempts = (retries or self.max_retries) + 1
        last_error = None

        for attempt in range(max_attempts):
            result = send_task(
                self.host,
                self.port,
                task_dict,
                timeout=self.timeout,
            )

            if not result.get("error"):
                self._send_count += 1
                return result

            last_error = result["error"]
            logger.warning(
                "Attempt %d/%d failed: %s",
                attempt + 1,
                max_attempts,
                last_error,
            )

        raise DistributedClientError(f"Failed after {max_attempts} attempts: {last_error}")

    @property
    def address(self) -> str:
        """Get worker address."""
        return f"{self.host}:{self.port}"

    @property
    def send_count(self) -> int:
        """Number of successful sends."""
        return self._send_count
