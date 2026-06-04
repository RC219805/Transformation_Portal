"""Bounded producer/consumer helpers for image batch I/O.

The pipeline overlaps three phases while preserving deterministic result order:
load in a background thread, process on the caller thread, and save in one or
more background threads. Callers provide the actual load/process/save functions
so this module remains independent of PIL, tifffile, and model runtimes.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Callable, Generic, Literal, Sequence, TypeVar, cast

LoadedT = TypeVar("LoadedT")
ProcessedT = TypeVar("ProcessedT")
SavedT = TypeVar("SavedT")

ParallelIOStage = Literal["load", "process", "save", "internal"]


@dataclass(frozen=True)
class ParallelIOItemResult(Generic[SavedT]):
    """Per-input result emitted by :class:`ParallelIOPipeline`."""

    input_path: Path
    output: SavedT | None = None
    error: Exception | None = None
    stage: ParallelIOStage | None = None

    @property
    def succeeded(self) -> bool:
        """Return True when load, process, and save all completed."""

        return self.error is None


@dataclass(frozen=True)
class _LoadedItem(Generic[LoadedT]):
    index: int
    input_path: Path
    payload: LoadedT


@dataclass(frozen=True)
class _SaveItem(Generic[ProcessedT]):
    index: int
    input_path: Path
    payload: ProcessedT


@dataclass(frozen=True)
class _Failure:
    index: int
    input_path: Path
    stage: ParallelIOStage
    error: Exception


_SENTINEL = object()


class ParallelIOPipeline(Generic[LoadedT, ProcessedT, SavedT]):
    """Overlap load, process, and save stages for bounded batch work."""

    def __init__(
        self,
        *,
        loader: Callable[[Path], LoadedT],
        saver: Callable[[Path, ProcessedT], SavedT],
        prefetch_size: int = 2,
        num_savers: int = 2,
        save_queue_size: int | None = None,
        thread_name_prefix: str = "parallel_io",
    ) -> None:
        if prefetch_size < 1:
            raise ValueError("prefetch_size must be >= 1")
        if num_savers < 1:
            raise ValueError("num_savers must be >= 1")
        if save_queue_size is not None and save_queue_size < 1:
            raise ValueError("save_queue_size must be >= 1 when provided")

        self._loader = loader
        self._saver = saver
        self.prefetch_size = prefetch_size
        self.num_savers = num_savers
        self.save_queue_size = save_queue_size or max(
            prefetch_size * num_savers,
            1,
        )
        self.thread_name_prefix = thread_name_prefix

    def process_batch(
        self,
        input_paths: Sequence[Path | str],
        processor_fn: Callable[[Path, LoadedT], ProcessedT],
    ) -> list[ParallelIOItemResult[SavedT]]:
        """Process *input_paths* while preserving caller order.

        Individual item failures are returned as failed results instead of
        aborting the entire batch. Unexpected worker failures still surface as
        ``internal`` item errors for inputs that never produced a result.
        """

        paths = [Path(path) for path in input_paths]
        if not paths:
            return []

        load_queue: Queue[_LoadedItem[LoadedT] | _Failure | object] = Queue(
            maxsize=self.prefetch_size,
        )
        save_queue: Queue[_SaveItem[ProcessedT] | object] = Queue(
            maxsize=self.save_queue_size,
        )
        results: list[ParallelIOItemResult[SavedT] | None] = [None] * len(paths)
        result_lock = threading.Lock()

        def record(index: int, result: ParallelIOItemResult[SavedT]) -> None:
            with result_lock:
                results[index] = result

        def loader_worker() -> None:
            try:
                for index, input_path in enumerate(paths):
                    try:
                        payload = self._loader(input_path)
                    except Exception as exc:
                        load_queue.put(
                            _Failure(index, input_path, "load", exc),
                        )
                    else:
                        load_queue.put(_LoadedItem(index, input_path, payload))
            finally:
                load_queue.put(_SENTINEL)

        def saver_worker() -> None:
            while True:
                item = save_queue.get()
                try:
                    if item is _SENTINEL:
                        return

                    save_item = cast(_SaveItem[ProcessedT], item)
                    try:
                        output = self._saver(
                            save_item.input_path,
                            save_item.payload,
                        )
                    except Exception as exc:
                        record(
                            save_item.index,
                            ParallelIOItemResult(
                                input_path=save_item.input_path,
                                error=exc,
                                stage="save",
                            ),
                        )
                    else:
                        record(
                            save_item.index,
                            ParallelIOItemResult(
                                input_path=save_item.input_path,
                                output=output,
                            ),
                        )
                finally:
                    save_queue.task_done()

        with ThreadPoolExecutor(
            max_workers=1 + self.num_savers,
            thread_name_prefix=self.thread_name_prefix,
        ) as executor:
            loader_future = executor.submit(loader_worker)
            saver_futures = [executor.submit(saver_worker) for _ in range(self.num_savers)]

            while True:
                loaded = load_queue.get()
                try:
                    if loaded is _SENTINEL:
                        break

                    if isinstance(loaded, _Failure):
                        record(
                            loaded.index,
                            ParallelIOItemResult(
                                input_path=loaded.input_path,
                                error=loaded.error,
                                stage=loaded.stage,
                            ),
                        )
                        continue

                    loaded_item = cast(_LoadedItem[LoadedT], loaded)
                    try:
                        processed = processor_fn(
                            loaded_item.input_path,
                            loaded_item.payload,
                        )
                    except Exception as exc:
                        record(
                            loaded_item.index,
                            ParallelIOItemResult(
                                input_path=loaded_item.input_path,
                                error=exc,
                                stage="process",
                            ),
                        )
                    else:
                        save_queue.put(
                            _SaveItem(
                                loaded_item.index,
                                loaded_item.input_path,
                                processed,
                            ),
                        )
                finally:
                    load_queue.task_done()

            loader_future.result()
            for _ in range(self.num_savers):
                save_queue.put(_SENTINEL)
            for saver_future in saver_futures:
                saver_future.result()

        return [
            (
                result
                if result is not None
                else ParallelIOItemResult(
                    input_path=paths[index],
                    error=RuntimeError(
                        "parallel I/O worker finished without producing a result",
                    ),
                    stage="internal",
                )
            )
            for index, result in enumerate(results)
        ]


__all__ = [
    "ParallelIOItemResult",
    "ParallelIOPipeline",
    "ParallelIOStage",
]
