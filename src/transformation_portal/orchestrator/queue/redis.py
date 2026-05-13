"""Redis-backed ``QueueBroker`` for multi-instance orchestrator deployments.

Phase 2.B parallel to Phase 1.B: the broker survives orchestrator /
worker restarts and lets a fleet of workers consume from a single
queue. Selectable via ``TP_ORCHESTRATOR_QUEUE_BACKEND=redis``
together with ``TP_REDIS_URL``.

Storage layout (all keys prefixed by ``key_prefix``, default ``tp:queue:``)::

    {prefix}ready                LIST   FIFO of job_ids ready for pickup.
    {prefix}leases               ZSET   member=job_id, score=deadline (Redis
                                        server seconds since epoch).
    {prefix}tracked              SET    every job_id the broker is currently
                                        responsible for (queued + leased).
    {prefix}job:<job_id>         HASH   dispatch payload + lease metadata.

Hash fields on ``{prefix}job:<job_id>``::

    request                JSON  ``JobEnqueueRequest`` payload, set on enqueue
                                 and never overwritten.
    worker_id              str   worker holding the lease (absent when queued).
    deadline               float Redis server time at which the lease expires.
    cancellation_requested '0' | '1'  Set to '1' by ``cancel`` for an
                                      in-flight job; surfaced via the next
                                      ``extend_lease`` as
                                      ``LeaseStatus.cancelled``.

Atomicity is implemented via server-side Lua so that admission
collisions, lease handoff, heartbeat extension, and reclaim each
happen in a single round-trip with no chance of partial state. Lease
deadlines are written from Redis ``TIME`` inside the acquire / extend
scripts. Production sweepers must read "now" back from the same
clock — call ``await broker.server_time()`` and pass the result to
``reclaim_expired_leases`` — so host wall-clock drift on individual
worker boxes cannot push leases into "expired" early or hold them
past their real deadline.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, List, Optional

from redis.asyncio import Redis

from transformation_portal.ingest.canonical_json import dumps_json
from transformation_portal.orchestrator.queue.base import (
    JobEnqueueRequest,
    JobLease,
    LeaseNotHeldError,
    LeaseStatus,
    QueueBroker,
    QueueBrokerError,
)

if TYPE_CHECKING:
    # ``AsyncScript`` lives in ``redis.commands.core`` today, but that
    # module is internal-ish — redis-py reorganizes it occasionally. Keep
    # the import behind ``TYPE_CHECKING`` so the broker module loads on
    # any redis-py 5.x / 6.x even if the path moves; at runtime we store
    # the script handles as ``Any`` (they're called as callables anyway).
    from redis.commands.core import AsyncScript  # noqa: F401

logger = logging.getLogger(__name__)


_DEFAULT_KEY_PREFIX = "tp:queue:"


# ---------------------------------------------------------------------------
# Lua scripts.
# Each script is documented with the keys it reads/writes and the argv shape
# so the call sites below are readable. Scripts return small payloads (string
# or array) rather than reaching into Python types from Lua.
# ---------------------------------------------------------------------------


# KEYS: ready, leases, tracked, job_hash
# ARGV: job_id, request_json
# Returns: 1 on success, 0 if the job_id was already tracked.
_LUA_ENQUEUE = """
if redis.call('SADD', KEYS[3], ARGV[1]) == 0 then
    return 0
end
redis.call('HSET', KEYS[4], 'request', ARGV[2])
redis.call('RPUSH', KEYS[1], ARGV[1])
return 1
"""


# KEYS: ready, leases, job_hash_prefix
# ARGV: worker_id, lease_seconds
# Returns: nil if empty, else {job_id, request_json, deadline}.
# Computes ``now`` from ``redis.call('TIME')`` so every worker resolves
# deadlines against the same clock.
_LUA_ACQUIRE = """
local job_id = redis.call('LPOP', KEYS[1])
if not job_id then
    return nil
end
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local deadline = now + tonumber(ARGV[2])
local job_key = KEYS[3] .. job_id
local request_json = redis.call('HGET', job_key, 'request')
redis.call('HSET', job_key, 'worker_id', ARGV[1], 'deadline', tostring(deadline))
redis.call('HDEL', job_key, 'cancellation_requested')
redis.call('ZADD', KEYS[2], deadline, job_id)
return {job_id, request_json, tostring(deadline)}
"""


# KEYS: leases, job_hash
# ARGV: worker_id, job_id, lease_seconds
# Returns:
#   'active'           lease extended; new deadline written.
#   'cancelled'        cancellation_requested == '1'; worker should release.
#   'not_held'         lease absent or held by a different worker.
_LUA_EXTEND = """
local lease_score = redis.call('ZSCORE', KEYS[1], ARGV[2])
if not lease_score then
    return 'not_held'
end
local holder = redis.call('HGET', KEYS[2], 'worker_id')
if holder ~= ARGV[1] then
    return 'not_held'
end
local cancel = redis.call('HGET', KEYS[2], 'cancellation_requested')
if cancel == '1' then
    return 'cancelled'
end
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local deadline = now + tonumber(ARGV[3])
redis.call('HSET', KEYS[2], 'deadline', tostring(deadline))
redis.call('ZADD', KEYS[1], deadline, ARGV[2])
return 'active'
"""


# KEYS: leases, tracked, job_hash
# ARGV: worker_id, job_id
# Returns: 1 if released, 0 if no-op (idempotent).
_LUA_RELEASE = """
local holder = redis.call('HGET', KEYS[3], 'worker_id')
if holder ~= ARGV[1] then
    return 0
end
redis.call('ZREM', KEYS[1], ARGV[2])
redis.call('SREM', KEYS[2], ARGV[2])
redis.call('DEL', KEYS[3])
return 1
"""


# KEYS: ready, leases, tracked, job_hash_prefix
# ARGV: now
# Returns: array of reclaimed job_ids.
# Re-queues each expired job at the head of ``ready`` so it is the
# next one picked up by a worker (matches MemoryQueueBroker semantics).
_LUA_RECLAIM = """
local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', ARGV[1])
for i = 1, #expired do
    local job_id = expired[i]
    local job_key = KEYS[4] .. job_id
    redis.call('ZREM', KEYS[2], job_id)
    redis.call('HDEL', job_key, 'worker_id', 'deadline')
    redis.call('LPUSH', KEYS[1], job_id)
end
return expired
"""


# KEYS: ready, leases, tracked, job_hash
# ARGV: job_id
# Returns:
#   'absent'     job_id not tracked; caller returns False.
#   'inflight'   marked cancellation_requested; caller returns True.
#   'queued'     removed from ready list; caller returns True.
_LUA_CANCEL = """
if redis.call('SISMEMBER', KEYS[3], ARGV[1]) == 0 then
    return 'absent'
end
local in_leases = redis.call('ZSCORE', KEYS[2], ARGV[1])
if in_leases then
    redis.call('HSET', KEYS[4], 'cancellation_requested', '1')
    return 'inflight'
end
redis.call('LREM', KEYS[1], 0, ARGV[1])
redis.call('SREM', KEYS[3], ARGV[1])
redis.call('DEL', KEYS[4])
return 'queued'
"""


class RedisQueueBroker(QueueBroker):
    """Multi-instance ``QueueBroker`` backed by Redis.

    A single Redis instance is assumed; ``redis_url`` follows the
    standard ``redis://[:password@]host:port/db`` form. The broker
    holds a connection pool for the lifetime of the instance and
    releases it on ``close``; ``run_worker_forever`` calls ``close``
    on shutdown.

    ``key_prefix`` lets parallel test runs (or shared-tenant Redis
    deployments) isolate their state without flushing the entire
    database. Production code typically accepts the default.
    """

    def __init__(
        self,
        *,
        redis_url: str,
        key_prefix: str = _DEFAULT_KEY_PREFIX,
        client: Optional[Redis] = None,
    ) -> None:
        if not key_prefix.endswith(":"):
            key_prefix = key_prefix + ":"
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        # Tests may inject a client (e.g. a fakeredis instance); production
        # constructs one from the URL.
        self._client: Redis = (
            client
            if client is not None
            else Redis.from_url(
                redis_url,
                decode_responses=True,
            )
        )
        self._owns_client = client is None
        self._scripts_registered = False
        self._enqueue_script: Optional[Any] = None
        self._acquire_script: Optional[Any] = None
        self._extend_script: Optional[Any] = None
        self._release_script: Optional[Any] = None
        self._reclaim_script: Optional[Any] = None
        self._cancel_script: Optional[Any] = None

    # ------------------------------------------------------------------ keys

    @property
    def _ready_key(self) -> str:
        return f"{self._key_prefix}ready"

    @property
    def _leases_key(self) -> str:
        return f"{self._key_prefix}leases"

    @property
    def _tracked_key(self) -> str:
        return f"{self._key_prefix}tracked"

    @property
    def _job_hash_prefix(self) -> str:
        return f"{self._key_prefix}job:"

    def _job_hash_key(self, job_id: str) -> str:
        return f"{self._job_hash_prefix}{job_id}"

    def _register_scripts(self) -> None:
        if self._scripts_registered:
            return
        self._enqueue_script = self._client.register_script(_LUA_ENQUEUE)
        self._acquire_script = self._client.register_script(_LUA_ACQUIRE)
        self._extend_script = self._client.register_script(_LUA_EXTEND)
        self._release_script = self._client.register_script(_LUA_RELEASE)
        self._reclaim_script = self._client.register_script(_LUA_RECLAIM)
        self._cancel_script = self._client.register_script(_LUA_CANCEL)
        self._scripts_registered = True

    # ------------------------------------------------------------------ API

    async def enqueue(self, request: JobEnqueueRequest) -> None:
        self._register_scripts()
        payload = dumps_json(
            {
                "job_id": request.job_id,
                "argv": request.argv,
                "api_version": request.api_version,
                "metadata": request.metadata,
            },
            sort_keys=True,
        )
        assert self._enqueue_script is not None
        added = await self._enqueue_script(
            keys=[self._ready_key, self._leases_key, self._tracked_key, self._job_hash_key(request.job_id)],
            args=[request.job_id, payload],
        )
        if int(added) == 0:
            raise QueueBrokerError(f"job {request.job_id!r} already pending in the queue or leased")

    async def acquire_lease(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> Optional[JobLease]:
        if lease_seconds <= 0:
            raise QueueBrokerError("lease_seconds must be positive")
        self._register_scripts()
        assert self._acquire_script is not None
        result = await self._acquire_script(
            keys=[self._ready_key, self._leases_key, self._job_hash_prefix],
            args=[worker_id, str(lease_seconds)],
        )
        if result is None:
            return None
        job_id, request_json, deadline_str = result
        if isinstance(job_id, bytes):
            job_id = job_id.decode("utf-8")
        if isinstance(request_json, bytes):
            request_json = request_json.decode("utf-8")
        if isinstance(deadline_str, bytes):
            deadline_str = deadline_str.decode("utf-8")
        request = _deserialize_request(request_json)
        deadline = float(deadline_str)
        return JobLease(
            job_id=job_id,
            worker_id=worker_id,
            deadline=deadline,
            request=request,
        )

    async def extend_lease(
        self,
        worker_id: str,
        job_id: str,
        *,
        lease_seconds: float,
    ) -> LeaseStatus:
        if lease_seconds <= 0:
            raise QueueBrokerError("lease_seconds must be positive")
        self._register_scripts()
        assert self._extend_script is not None
        result = await self._extend_script(
            keys=[self._leases_key, self._job_hash_key(job_id)],
            args=[worker_id, job_id, str(lease_seconds)],
        )
        outcome = _decode(result)
        if outcome == "not_held":
            raise LeaseNotHeldError(worker_id=worker_id, job_id=job_id)
        if outcome == "cancelled":
            return LeaseStatus.cancelled
        if outcome == "active":
            return LeaseStatus.active
        raise QueueBrokerError(f"unexpected extend_lease outcome from Redis: {outcome!r}")

    async def release_lease(self, worker_id: str, job_id: str) -> None:
        self._register_scripts()
        assert self._release_script is not None
        await self._release_script(
            keys=[self._leases_key, self._tracked_key, self._job_hash_key(job_id)],
            args=[worker_id, job_id],
        )

    async def reclaim_expired_leases(self, *, now: float) -> List[str]:
        self._register_scripts()
        assert self._reclaim_script is not None
        result = await self._reclaim_script(
            keys=[self._ready_key, self._leases_key, self._tracked_key, self._job_hash_prefix],
            args=[str(now)],
        )
        return [_decode(item) for item in result]

    async def cancel(self, job_id: str) -> bool:
        self._register_scripts()
        assert self._cancel_script is not None
        result = await self._cancel_script(
            keys=[self._ready_key, self._leases_key, self._tracked_key, self._job_hash_key(job_id)],
            args=[job_id],
        )
        outcome = _decode(result)
        if outcome == "absent":
            return False
        if outcome in ("inflight", "queued"):
            return True
        raise QueueBrokerError(f"unexpected cancel outcome from Redis: {outcome!r}")

    async def queued_job_ids(self) -> List[str]:
        items = await self._client.lrange(self._ready_key, 0, -1)
        return [_decode(item) for item in items]

    async def leased_job_ids(self) -> List[str]:
        items = await self._client.zrange(self._leases_key, 0, -1)
        return [_decode(item) for item in items]

    async def server_time(self) -> float:
        """Read the Redis server clock — the broker's source of truth.

        Lease deadlines are written from Redis ``TIME`` inside the
        acquire / extend Lua scripts, so production sweepers must
        compare against this same clock to defeat host wall-clock
        drift in a multi-host fleet. Returns seconds since the Unix
        epoch as a float (microsecond resolution from Redis).
        """
        seconds, microseconds = await self._client.time()
        return float(seconds) + float(microseconds) / 1_000_000

    async def reset(self) -> None:
        """Test-only: drop every key under ``key_prefix``.

        Uses ``SCAN`` (not ``KEYS``) so we never block Redis on a
        shared instance, and never touches keys outside the broker's
        prefix — multiple parametrized test runs can share a Redis
        with different ``key_prefix`` values.
        """
        pattern = f"{self._key_prefix}*"
        cursor = 0
        first = True
        while first or cursor != 0:
            first = False
            cursor, keys = await self._client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                await self._client.delete(*keys)

    async def close(self) -> None:
        if not self._owns_client:
            return
        try:
            await self._client.aclose()
        except AttributeError:
            # redis-py < 5.0 used ``close`` + ``connection_pool.disconnect``.
            await self._client.close()  # type: ignore[func-returns-value]
            await self._client.connection_pool.disconnect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value) if value is not None else ""


def _deserialize_request(payload: str) -> JobEnqueueRequest:
    data = json.loads(payload)
    return JobEnqueueRequest(
        job_id=data["job_id"],
        argv=list(data["argv"]),
        api_version=data.get("api_version", "v1"),
        metadata=dict(data.get("metadata") or {}),
    )


__all__ = ["RedisQueueBroker"]
