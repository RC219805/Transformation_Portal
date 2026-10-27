"""Process-local, bounded admission for photographic review work.

These are accounting reservations, not operating-system memory/disk reservations.
The opt-in pilot requires one backend process. Unrelated filesystem writers and
other processes are outside this controller's authority.
"""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import stat
import tempfile
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

GIB = 1024**3
MIB = 1024**2
_Result = TypeVar("_Result")


class ReviewOverload(ValueError):
    """Public, typed admission rejection; never includes host or tenant details."""

    def __init__(self, reason: str, *, status: int = 503) -> None:
        super().__init__(
            "Photographic capacity is unavailable; retry later"
            if status != 413
            else "Photographic operation exceeds the configured capacity"
        )
        self.reason, self.status = reason, status
        self.retriable = status != 413


@dataclass(frozen=True)
class ReviewAdmissionConfig:
    enabled: bool = False
    memory_bytes: int = 12 * GIB
    interactive_memory_bytes: int = 4 * GIB
    bulk_memory_bytes: int = 8 * GIB
    decode_slots: int = 2
    bulk_decode_slots: int = 1
    io_slots: int = 2
    interactive_io_slots: int = 1
    noninteractive_slots: int = 1
    tenant_slots: int = 2
    tenant_noninteractive_slots: int = 1
    outstanding: int = 32
    tenant_outstanding: int = 8
    queued: int = 24
    tenant_queued: int = 6
    interactive_wait_seconds: int = 5
    bulk_wait_seconds: int = 30
    scratch_bytes: int = 64 * GIB
    interactive_scratch_bytes: int = 2 * GIB
    scratch_floor_bytes: int = 2 * GIB

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "enabled":
                if type(value) is not bool:
                    raise ValueError("Photographic admission enabled must be boolean")
            else:
                maximum = (
                    1024 * GIB
                    if name.endswith("_bytes")
                    else 300 if name.endswith("_seconds") else 4096 if name.endswith(("outstanding", "queued")) else 64
                )
                if type(value) is not int or not 0 < value <= maximum:
                    raise ValueError(f"Photographic admission {name} must be a bounded positive integer")
        if (
            self.bulk_memory_bytes > self.memory_bytes - self.interactive_memory_bytes
            or self.bulk_decode_slots >= self.decode_slots
            or self.interactive_io_slots >= self.io_slots
            or self.noninteractive_slots > self.io_slots - self.interactive_io_slots
            or self.tenant_noninteractive_slots >= self.tenant_slots
            or self.interactive_scratch_bytes >= self.scratch_bytes
            or self.tenant_outstanding > self.outstanding
            or self.queued >= self.outstanding
            or self.tenant_queued >= self.tenant_outstanding
            or self.tenant_queued > self.queued
        ):
            raise ValueError("Photographic admission limits do not preserve interactive capacity")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> ReviewAdmissionConfig:
        values = os.environ if environment is None else environment
        supplied: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            raw = values.get("TP_PHOTO_REVIEW_" + name.upper())
            if raw is None:
                continue
            if name == "enabled":
                if raw not in {"0", "1"}:
                    raise ValueError("TP_PHOTO_REVIEW_ENABLED must be 0 or 1")
                supplied[name] = raw == "1"
            else:
                if not raw.isascii() or not raw.isdecimal() or len(raw) > 15:
                    raise ValueError(f"Invalid TP_PHOTO_REVIEW_{name.upper()}")
                supplied[name] = int(raw)
        return cls(**supplied)


@dataclass(frozen=True)
class ReviewCost:
    memory_bytes: int
    scratch_bytes: int
    decode_slots: int = 1
    io_slots: int = 1
    interactive: bool = True
    max_pixels: int = 40_000_000
    filesystems: tuple[tuple[Path, int], ...] = ()

    def __post_init__(self) -> None:
        if type(self.interactive) is not bool or any(
            type(value) is not int or value < 0
            for value in (self.memory_bytes, self.scratch_bytes, self.decode_slots, self.io_slots, self.max_pixels)
        ):
            raise ValueError("Photographic costs require nonnegative integer reservations")
        if self.filesystems and (
            sum(size for _, size in self.filesystems) != self.scratch_bytes
            or any(type(size) is not int or size < 0 or not root.is_absolute() for root, size in self.filesystems)
        ):
            raise ValueError("Scratch filesystem claims must match the complete scratch reservation")


# Foundation CPU probes at f79f49207ed2e4589b006017dec90edd51137cb2.
# Incremental process high-water RSS, including conservative import peaks.
# Also includes recorded-path candidate diff 9457bced028e600360b5c55d4fb7552c7b7cb9b69a37f254496f109f7f73a86d.
# Final recorded-path probes bind math commit 3f1f3a61da03855ff8b3a37dc75545e0c244da57
# at report HEAD 2f6faa712a6aea3e3b8cf82b1f32de26fe0928d5, with all eight source
# hashes verified (FINAL_RECORDED_RESOURCE_PROFILES.json, 25 cases). Preserve
# the higher observed baseline/candidate peaks below; one lower run cannot
# justify reducing reservations. These are CPU allocation measurements only.
# Cases: 40M RGB float32 TIFF and 40M descriptor-bound native RGB + alpha.
# Failure of transparent SDR export is a contract rejection, not a zero cost.
_MEASURED_40MP = {
    "metadata": (3_003_842_560, 1_480_851_456),
    "render": (3_507_388_416, 2_944_270_336),
    "revision": (3_987_947_520, 6_153_633_792),
    "delivery": (3_734_503_424, 3_734_503_424),
}
MAX_REVIEW_PIXELS = 40_000_000


def effective_interactive_reserve(config: ReviewAdmissionConfig) -> int:
    # Whole-GiB rounding plus one GiB allocator/codec headroom. This is a
    # configured reservation, not an automatic expansion of the process cap.
    measured = math.ceil(max(_MEASURED_40MP["render"]) * 1.5 / GIB) * GIB + GIB
    return max(config.interactive_memory_bytes, measured)


def operation_cost(
    operation: str, config: ReviewAdmissionConfig, inspections: list[dict[str, Any]] | None = None
) -> ReviewCost:
    """Reserve max(analytical allocation bound, 1.5 * observed peak).

    Only server-recorded hash-bound inspections may reduce the 40M-pixel
    fallback. Verified image headers are checked against this bound before
    decoding. Missing/unknown fields choose the conservative profile; they do
    not infer a processing stage from pixel values or names.
    """
    if operation == "download":
        return ReviewCost(256 * MIB, 0, decode_slots=0, interactive=False)
    if operation not in {"render", "decision", "catalogue", "revision", "delivery", "quality", "groups"}:
        raise ValueError("Unknown photographic operation")
    pixels = MAX_REVIEW_PIXELS
    native_alpha = True
    if inspections and all(
        type(row.get("width")) is int
        and type(row.get("height")) is int
        and 0 < row["width"] * row["height"] <= MAX_REVIEW_PIXELS
        and min(row["width"], row["height"]) > 0
        and type(row.get("alpha")) is bool
        for row in inspections
    ):
        pixels = max(row["width"] * row["height"] for row in inspections)
        native_alpha = any(row["alpha"] for row in inspections)
    profile = "metadata" if operation in {"decision", "catalogue"} else operation
    # Native/encoded source snapshots, ICC/descriptor copies and output buffers.
    snapshots = 2 * (512 * MIB + 18 * MIB) + 256 * MIB
    coefficients = {"metadata": 64, "render": 64, "revision": 144, "delivery": 128, "quality": 192, "groups": 128}
    analytical = pixels * coefficients[profile] + snapshots
    observations = _MEASURED_40MP.get(profile)
    measured = 0
    if observations is not None:
        measured = math.ceil((max(observations) if native_alpha else observations[0]) * 1.5 * pixels / MAX_REVIEW_PIXELS)
    memory = math.ceil(max(analytical, measured) / MIB) * MIB
    interactive = operation in {"render", "decision", "catalogue"}
    return ReviewCost(
        memory,
        0,  # The service supplies geometry-derived filesystem claims, never a lane-sized placeholder.
        interactive=interactive,
        max_pixels=pixels,
    )


async def drain_owned(awaitable: Awaitable[_Result]) -> _Result:
    """Cancellation never abandons a worker or a reservation release."""
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


@dataclass(eq=False)
class ReviewLease:
    controller: PhotographicAdmissionController
    tenant_id: str
    cost: ReviewCost
    released: bool = False
    tracked_paths: set[Path] = field(default_factory=set)
    future_scratch_finished: bool = False

    async def release_expensive(self) -> None:
        """Workers/publication are quiescent; CAS and cleanup own scratch only."""
        await drain_owned(self.controller._release_expensive(self))

    def track_path(self, path: Path) -> None:
        self.tracked_paths.add(path)

    def claims(self) -> tuple[tuple[Path, int], ...]:
        return self.cost.filesystems or ((Path(tempfile.gettempdir()).resolve(), self.cost.scratch_bytes),)

    def remaining(self, device: int) -> int:
        if self.future_scratch_finished:
            return 0
        claimed = sum(size for root, size in self.claims() if root.stat().st_dev == device)
        observed = 0
        seen: set[tuple[int, int]] = set()
        count = 0
        for root in self.tracked_paths:
            paths = [root]
            while paths:
                path = paths.pop()
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                count += 1
                if count > 4096:
                    return claimed  # Unknown footprint never grants extra capacity.
                if info.st_dev != device or stat.S_ISLNK(info.st_mode):
                    continue
                if stat.S_ISDIR(info.st_mode):
                    try:
                        paths.extend(path.iterdir())
                    except FileNotFoundError:
                        # A response cleanup worker can remove its private
                        # directory after lstat while its lease is still owned.
                        # An incomplete footprint grants no allocation credit.
                        return claimed
                elif stat.S_ISREG(info.st_mode) and (device, info.st_ino) not in seen:
                    seen.add((device, info.st_ino))
                    observed += info.st_blocks * 512
        return max(0, claimed - observed)

    async def release(self) -> None:
        await drain_owned(self.controller._release(self))

    def check_scratch(self, required_bytes: int, *, preparing: bool = False) -> None:
        """Check every affected filesystem against all remaining commitments.

        Tracked private allocations are already reflected in physical free
        space, so only each owner's remaining commitment is added. Never count
        existing retained namespaces as owned allocations.
        """
        if type(required_bytes) is not int or required_bytes < 0 or required_bytes > self.cost.scratch_bytes:
            raise ReviewOverload("operation_scratch_limit", status=413)
        config = self.controller.config
        devices: dict[int, Path] = {}
        for owner in self.controller._active:
            for root, _ in owner.claims():
                devices[root.stat().st_dev] = root
        temporary_device = Path(tempfile.gettempdir()).stat().st_dev
        for device, root in devices.items():
            remaining = sum(owner.remaining(device) for owner in self.controller._active)
            interactive_claims = sum(
                size
                for owner in self.controller._active
                if owner.cost.interactive and not owner.future_scratch_finished
                for path, size in owner.claims()
                if path.stat().st_dev == device
            )
            protected = (
                max(0, config.interactive_scratch_bytes - interactive_claims)
                if device == temporary_device
                and any(not owner.cost.interactive and not owner.future_scratch_finished for owner in self.controller._active)
                else 0
            )
            if shutil.disk_usage(root).free < remaining + protected + config.scratch_floor_bytes:
                raise ReviewOverload("scratch_unavailable")


current_review_lease: ContextVar[ReviewLease | None] = ContextVar("current_review_lease", default=None)


@contextmanager
def owned_review_directory(**kwargs: Any) -> Iterator[str]:
    """Track only this operation's private scratch, through context cleanup."""
    with tempfile.TemporaryDirectory(**kwargs) as directory:
        lease = current_review_lease.get()
        path = Path(directory)
        if lease is not None:
            lease.track_path(path)
        try:
            yield directory
        finally:
            if lease is not None:
                lease.tracked_paths.discard(path)


@dataclass(eq=False)
class ReviewRequestTicket:
    controller: PhotographicAdmissionController
    tenant_id: str
    released: bool = False

    async def release(self) -> None:
        await drain_owned(self.controller._release_request_ticket(self))


async def run_review_thread(operation: Callable[..., _Result], *args: Any, **kwargs: Any) -> _Result:
    lease = current_review_lease.get()
    if lease is None:
        return await asyncio.to_thread(operation, *args, **kwargs)
    return await lease.controller.run_blocking(operation, *args, **kwargs)


@dataclass(eq=False)
class _Waiter:
    tenant_id: str
    cost: ReviewCost
    future: asyncio.Future[ReviewLease]


@dataclass(eq=False)
class _RenderGroup:
    task: asyncio.Task[Any] | None = None
    subscribers: int = 0
    started: bool = False


_current_render_group: ContextVar[_RenderGroup | None] = ContextVar("current_render_group", default=None)


class PhotographicAdmissionController:
    """FIFO within tenant/class, round-robin across tenants, bounded waiters."""

    def __init__(self, config: ReviewAdmissionConfig | None = None) -> None:
        self.config = config if config is not None else ReviewAdmissionConfig.from_environment()
        self.interactive_reserve = effective_interactive_reserve(self.config)
        if self.config.enabled and self.interactive_reserve >= self.config.memory_bytes:
            raise ValueError("Photographic memory ceiling cannot preserve measured inspection capacity")
        self._lock = asyncio.Lock()
        self._closed = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._executor: ThreadPoolExecutor | None = None
        self._requests: dict[str, int] = {}
        self._authorizing = 0
        self._active: list[ReviewLease] = []
        self._queues: dict[bool, dict[str, deque[_Waiter]]] = {True: {}, False: {}}
        self._tenants: dict[bool, deque[str]] = {True: deque(), False: deque()}
        self._interactive_grants = 0
        self._groups: dict[tuple[Any, ...], _RenderGroup] = {}
        self._statistics: dict[str, int | float] = {"admitted": 0, "rejected": 0, "wait_seconds": 0.0, "coalesced": 0}

    async def run_blocking(self, operation: Callable[..., _Result], *args: Any, **kwargs: Any) -> _Result:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max(self.config.decode_slots, self.config.io_slots), thread_name_prefix="tp-photo"
            )
        context = copy_context()
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, partial(context.run, operation, *args, **kwargs)
        )

    async def close(self) -> None:
        """Stop admission, fail waiters, drain owners, then stop the bounded pool."""
        async with self._lock:
            self._closed = True
            for queues in self._queues.values():
                for queue in queues.values():
                    for waiter in queue:
                        waiter.future.set_exception(ReviewOverload("service_stopping"))
                queues.clear()
            for tenants in self._tenants.values():
                tenants.clear()
        await drain_owned(self._idle.wait())
        if self._executor is not None:
            await drain_owned(asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=False))

    def snapshot(self) -> dict[str, Any]:
        """Bounded process-level diagnostics; no actors, source keys or tenants."""
        return {
            "scope": "process",
            "enabled": self.config.enabled,
            "interactive_memory_protected_bytes": self.interactive_reserve,
            "outstanding": sum(self._requests.values()) + self._authorizing,
            "queued": self._queue_count(),
            "active": len(self._active),
            "memory_reserved_bytes": sum(row.cost.memory_bytes for row in self._active),
            "scratch_reserved_bytes": sum(row.cost.scratch_bytes for row in self._active),
            **self._statistics,
        }

    @asynccontextmanager
    async def authorization(self) -> AsyncIterator[None]:
        """Bound lightweight auth without trusting a client-selected tenant."""
        async with self._lock:
            if self._closed or sum(self._requests.values()) + self._authorizing >= self.config.outstanding:
                raise ReviewOverload("process_outstanding_limit")
            self._authorizing += 1
        try:
            yield
        finally:
            await drain_owned(self._release_authorization())

    async def _release_authorization(self) -> None:
        async with self._lock:
            self._authorizing -= 1

    async def begin_request(self, tenant_id: str) -> ReviewRequestTicket:
        async with self._lock:
            if self._requests.get(tenant_id, 0) >= self.config.tenant_outstanding:
                self._statistics["rejected"] += 1
                raise ReviewOverload("tenant_outstanding_limit", status=429)
            if sum(self._requests.values()) + self._authorizing >= self.config.outstanding:
                self._statistics["rejected"] += 1
                raise ReviewOverload("process_outstanding_limit")
            self._requests[tenant_id] = self._requests.get(tenant_id, 0) + 1
        return ReviewRequestTicket(self, tenant_id)

    @asynccontextmanager
    async def request(self, tenant_id: str) -> AsyncIterator[None]:
        ticket = await self.begin_request(tenant_id)
        try:
            yield
        finally:
            await ticket.release()

    async def _release_request_ticket(self, ticket: ReviewRequestTicket) -> None:
        async with self._lock:
            if ticket.released:
                return
            remaining = self._requests[ticket.tenant_id] - 1
            if remaining:
                self._requests[ticket.tenant_id] = remaining
            else:
                del self._requests[ticket.tenant_id]
            ticket.released = True

    def _queue_count(self, tenant_id: str | None = None) -> int:
        return sum(
            len(queue)
            for queues in self._queues.values()
            for tenant, queue in queues.items()
            if tenant_id is None or tenant == tenant_id
        )

    def _validate_cost(self, cost: ReviewCost) -> None:
        config = self.config
        if (
            cost.memory_bytes
            > (
                config.memory_bytes
                if cost.interactive
                else min(config.bulk_memory_bytes, config.memory_bytes - self.interactive_reserve)
            )
            or cost.scratch_bytes > config.scratch_bytes - (0 if cost.interactive else config.interactive_scratch_bytes)
            or cost.decode_slots > (config.decode_slots if cost.interactive else config.bulk_decode_slots)
            or cost.io_slots > (config.io_slots if cost.interactive else config.noninteractive_slots)
        ):
            raise ReviewOverload("operation_capacity_limit", status=413)

    def _fits(self, waiter: _Waiter) -> bool:
        config, cost = self.config, waiter.cost
        bulk = [row for row in self._active if not row.cost.interactive and row.cost.io_slots]
        tenant = [row for row in self._active if row.tenant_id == waiter.tenant_id and row.cost.io_slots]
        if (
            len(tenant) >= config.tenant_slots
            or sum(row.cost.memory_bytes for row in self._active) + cost.memory_bytes > config.memory_bytes
            or sum(row.cost.scratch_bytes for row in self._active) + cost.scratch_bytes > config.scratch_bytes
            or sum(row.cost.decode_slots for row in self._active) + cost.decode_slots > config.decode_slots
            or sum(row.cost.io_slots for row in self._active) + cost.io_slots > config.io_slots
        ):
            return False
        if not cost.interactive and (
            len(bulk) >= config.noninteractive_slots
            or sum(not row.cost.interactive for row in tenant) >= config.tenant_noninteractive_slots
            or sum(row.cost.memory_bytes for row in bulk) + cost.memory_bytes > config.memory_bytes - self.interactive_reserve
            # Publication keeps its scratch commitment after releasing workers.
            # That retained claim must still protect inspection headroom.
            or sum(row.cost.scratch_bytes for row in self._active if not row.cost.interactive) + cost.scratch_bytes
            > config.scratch_bytes - config.interactive_scratch_bytes
            or sum(row.cost.decode_slots for row in bulk) + cost.decode_slots > config.bulk_decode_slots
            or sum(row.cost.io_slots for row in bulk) + cost.io_slots > config.io_slots - config.interactive_io_slots
        ):
            return False
        return True

    def _grant(self, waiter: _Waiter) -> None:
        lease = ReviewLease(self, waiter.tenant_id, waiter.cost)
        self._active.append(lease)
        self._idle.clear()
        self._statistics["admitted"] += 1
        waiter.future.set_result(lease)
        self._interactive_grants = self._interactive_grants + 1 if waiter.cost.interactive else 0

    def _schedule(self) -> None:
        if self._closed:
            return
        while True:
            order = (False, True) if self._interactive_grants >= 8 else (True, False)
            granted = False
            for interactive in order:
                tenants, queues = self._tenants[interactive], self._queues[interactive]
                for _ in range(len(tenants)):
                    tenant = tenants.popleft()
                    queue = queues[tenant]
                    waiter = queue[0]
                    if not self._fits(waiter):
                        tenants.append(tenant)
                        continue
                    queue.popleft()
                    if queue:
                        tenants.append(tenant)
                    else:
                        del queues[tenant]
                    self._grant(waiter)
                    granted = True
                    break
                if granted:
                    break
            if not granted:
                return

    async def acquire(self, tenant_id: str, cost: ReviewCost) -> ReviewLease:
        self._validate_cost(cost)
        waiter = _Waiter(tenant_id, cost, asyncio.get_running_loop().create_future())
        started = time.monotonic()
        async with self._lock:
            if self._closed:
                raise ReviewOverload("service_stopping")
            queues, tenants = self._queues[cost.interactive], self._tenants[cost.interactive]
            self._schedule()
            # Queue caps bound actual waiters. Do not reject an inspector that
            # can immediately use protected capacity merely because bulk work
            # has filled the waiting queue. Older eligible/FIFO work runs first.
            if not queues.get(tenant_id) and self._fits(waiter):
                self._grant(waiter)
            else:
                if self._queue_count(tenant_id) >= self.config.tenant_queued:
                    raise ReviewOverload("tenant_queue_limit", status=429)
                if self._queue_count() >= self.config.queued:
                    raise ReviewOverload("process_queue_limit")
                if tenant_id not in queues:
                    queues[tenant_id] = deque()
                    tenants.append(tenant_id)
                queues[tenant_id].append(waiter)
                self._schedule()
        timeout = self.config.interactive_wait_seconds if cost.interactive else self.config.bulk_wait_seconds
        try:
            lease = await asyncio.wait_for(asyncio.shield(waiter.future), timeout)
        except BaseException as error:
            await drain_owned(self._remove_waiter(waiter))
            if isinstance(error, TimeoutError):
                self._statistics["rejected"] += 1
                raise ReviewOverload("admission_deadline") from error
            raise
        self._statistics["wait_seconds"] += time.monotonic() - started
        group = _current_render_group.get()
        if group is not None:
            group.started = True
        return lease

    async def _remove_waiter(self, waiter: _Waiter) -> None:
        async with self._lock:
            if waiter.future.done() and not waiter.future.cancelled() and waiter.future.exception() is None:
                lease = waiter.future.result()
                self._active.remove(lease)
                lease.released = True
            else:
                queues, tenants = self._queues[waiter.cost.interactive], self._tenants[waiter.cost.interactive]
                queue = queues.get(waiter.tenant_id)
                if queue is not None and waiter in queue:
                    queue.remove(waiter)
                    if not queue:
                        del queues[waiter.tenant_id]
                        tenants.remove(waiter.tenant_id)
                waiter.future.cancel()
            if not self._active:
                self._idle.set()
            self._schedule()

    async def _release_expensive(self, lease: ReviewLease) -> None:
        if lease.released or lease.future_scratch_finished:
            return
        # The worker and publication are quiescent. Keep only private bytes
        # still owned through CAS/rollback, not the completed future workspace.
        devices: dict[int, tuple[Path, int]] = {}
        for root, size in lease.claims():
            device = root.stat().st_dev
            devices[device] = (root, devices.get(device, (root, 0))[1] + size)
        held = tuple((root, max(0, size - lease.remaining(device))) for device, (root, size) in devices.items())
        async with self._lock:
            if not lease.released and not lease.future_scratch_finished:
                lease.cost = replace(
                    lease.cost,
                    memory_bytes=0,
                    decode_slots=0,
                    io_slots=0,
                    scratch_bytes=sum(size for _, size in held),
                    filesystems=held,
                )
                lease.future_scratch_finished = True
                self._schedule()

    async def _release(self, lease: ReviewLease) -> None:
        async with self._lock:
            if not lease.released:
                self._active.remove(lease)
                lease.released = True
                if not self._active:
                    self._idle.set()
                self._schedule()

    async def coalesce(self, key: tuple[Any, ...], operation: Callable[[], Awaitable[_Result]]) -> _Result:
        """Share in-flight reads, never results or caller authorization.

        Last-interest cancellation removes a queued group. Running work drains
        under its own lease; cancelling one subscriber cannot cancel another.
        """
        async with self._lock:
            group = self._groups.get(key)
            if group is None or group.task is None or group.task.done():
                group = _RenderGroup()

                async def run() -> _Result:
                    token = _current_render_group.set(group)
                    try:
                        return await operation()
                    finally:
                        _current_render_group.reset(token)

                group.task = asyncio.create_task(run())
                self._groups[key] = group
                group.task.add_done_callback(lambda done: self._forget_group(key, group, done))
            else:
                self._statistics["coalesced"] += 1
            group.subscribers += 1
        try:
            return await asyncio.shield(group.task)
        finally:
            await drain_owned(self._detach_group(key, group))

    async def _detach_group(self, key: tuple[Any, ...], group: _RenderGroup) -> None:
        async with self._lock:
            group.subscribers -= 1
            if group.subscribers == 0 and not group.started and group.task is not None and not group.task.done():
                group.task.cancel()
                if self._groups.get(key) is group:
                    del self._groups[key]

    def _forget_group(self, key: tuple[Any, ...], group: _RenderGroup, task: asyncio.Task[Any]) -> None:
        if self._groups.get(key) is group:
            del self._groups[key]
        if not task.cancelled():
            task.exception()
