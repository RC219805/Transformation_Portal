"""Versioned Redis dispatch-locator queue, isolated from legacy argv queues."""

from __future__ import annotations

from typing import Any

from transformation_portal.orchestrator.dispatch import DispatchLocator
from transformation_portal.orchestrator.queue.base import JobEnqueueRequest, QueueBrokerError
from transformation_portal.orchestrator.queue.redis import _LUA_EXTEND, RedisQueueBroker

# Expired dispatches are quarantined, never placed back in the ready list.
# The Postgres expiry reconciler owns tombstoning and admission release.
_RECLAIM_LOCATORS = """
local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', ARGV[1], 'LIMIT', 0, 1000)
for i = 1, #expired do
    local job_id = expired[i]
    local job_key = KEYS[4] .. job_id
    redis.call('ZREM', KEYS[2], job_id)
    redis.call('LREM', KEYS[1], 0, job_id)
    redis.call('SREM', KEYS[3], job_id)
    redis.call('DEL', job_key)
end
return expired
"""

_EXTEND_LOCATORS = _LUA_EXTEND.replace(
    "local deadline = now + tonumber(ARGV[3])",
    "if now >= tonumber(lease_score) then return 'not_held' end\nlocal deadline = now + tonumber(ARGV[3])",
)


class RedisLocatorQueueBroker(RedisQueueBroker):
    """Only closed locators; a separate namespace is mandatory for cutover."""

    def __init__(self, *, redis_url: str, key_prefix: str = "tp:dispatch:v1:", **kwargs: Any) -> None:
        if not key_prefix.endswith(":dispatch:v1:"):
            raise QueueBrokerError("locator queues require an isolated :dispatch:v1: suffix")
        super().__init__(redis_url=redis_url, key_prefix=key_prefix, **kwargs)
        self._locator_scripts_registered = False

    def _register_scripts(self) -> None:
        super()._register_scripts()
        if not self._locator_scripts_registered:
            self._enqueue_script = self._client.register_script("""
if redis.call('SISMEMBER', KEYS[3], ARGV[1]) == 1 then
    if redis.call('HGET', KEYS[4], 'request') == ARGV[2] then return 2 end
    return 0
end
redis.call('SADD', KEYS[3], ARGV[1])
redis.call('HSET', KEYS[4], 'request', ARGV[2])
redis.call('RPUSH', KEYS[1], ARGV[1])
return 1
""")
            self._reclaim_script = self._client.register_script(_RECLAIM_LOCATORS)
            self._extend_script = self._client.register_script(_EXTEND_LOCATORS)
            self._locator_scripts_registered = True

    @staticmethod
    def _serialize_request(request: JobEnqueueRequest | DispatchLocator) -> str:
        if not isinstance(request, DispatchLocator):
            raise QueueBrokerError("locator queue rejects raw executable requests")
        return request.to_json()

    @staticmethod
    def _parse_request(payload: str) -> DispatchLocator:
        try:
            return DispatchLocator.from_json(payload)
        except (TypeError, ValueError) as exc:
            raise QueueBrokerError("invalid dispatch locator") from exc
