from __future__ import annotations

import threading
from pathlib import Path

import pytest

from transformation_portal.pipelines.parallel_io import ParallelIOPipeline

pytestmark = pytest.mark.unit


def test_parallel_io_preserves_input_order_and_writes_outputs(
    tmp_path: Path,
) -> None:
    input_paths = [tmp_path / f"input_{index}.txt" for index in range(3)]
    for index, input_path in enumerate(input_paths):
        input_path.write_text(str(index), encoding="utf-8")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def loader(path: Path) -> int:
        return int(path.read_text(encoding="utf-8"))

    def processor(path: Path, value: int) -> tuple[Path, int]:
        return output_dir / path.name, value * 10

    def saver(_path: Path, payload: tuple[Path, int]) -> Path:
        output_path, value = payload
        output_path.write_text(str(value), encoding="utf-8")
        return output_path

    pipeline = ParallelIOPipeline(
        loader=loader,
        saver=saver,
        prefetch_size=2,
        num_savers=2,
    )

    results = pipeline.process_batch(input_paths, processor)

    assert [result.input_path for result in results] == input_paths
    assert [result.succeeded for result in results] == [True, True, True]
    assert [result.output.name for result in results if result.output is not None] == [path.name for path in input_paths]
    assert [(output_dir / path.name).read_text(encoding="utf-8") for path in input_paths] == ["0", "10", "20"]


def test_parallel_io_reports_load_process_and_save_failures(
    tmp_path: Path,
) -> None:
    input_paths = [
        tmp_path / "load_fail.txt",
        tmp_path / "process_fail.txt",
        tmp_path / "save_fail.txt",
        tmp_path / "ok.txt",
    ]
    for input_path in input_paths:
        input_path.write_text(input_path.stem, encoding="utf-8")

    def loader(path: Path) -> str:
        if path.name == "load_fail.txt":
            raise OSError("load failed")
        return path.read_text(encoding="utf-8")

    def processor(path: Path, value: str) -> str:
        if path.name == "process_fail.txt":
            raise RuntimeError("process failed")
        return value.upper()

    def saver(path: Path, value: str) -> str:
        if path.name == "save_fail.txt":
            raise ValueError("save failed")
        return f"{path.name}:{value}"

    pipeline = ParallelIOPipeline(
        loader=loader,
        saver=saver,
        prefetch_size=2,
        num_savers=2,
    )

    results = pipeline.process_batch(input_paths, processor)

    assert [(result.input_path.name, result.succeeded, result.stage) for result in results] == [
        ("load_fail.txt", False, "load"),
        ("process_fail.txt", False, "process"),
        ("save_fail.txt", False, "save"),
        ("ok.txt", True, None),
    ]
    assert results[-1].output == "ok.txt:OK"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefetch_size": 0},
        {"num_savers": 0},
        {"save_queue_size": 0},
    ],
)
def test_parallel_io_rejects_invalid_bounds(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ParallelIOPipeline(
            loader=lambda path: path,
            saver=lambda _path, payload: payload,
            **kwargs,
        )


def test_parallel_io_overlaps_processing_with_background_save(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    first_save_started = threading.Event()
    release_first_save = threading.Event()
    events: list[str] = []
    events_lock = threading.Lock()

    def append_event(value: str) -> None:
        with events_lock:
            events.append(value)

    def loader(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def processor(path: Path, value: str) -> str:
        if path == second:
            assert first_save_started.wait(timeout=2.0)
            append_event("process-second")
            release_first_save.set()
        return value

    def saver(path: Path, value: str) -> str:
        if path == first:
            append_event("save-first-start")
            first_save_started.set()
            assert release_first_save.wait(timeout=2.0)
        append_event(f"save-{path.stem}-end")
        return value

    pipeline = ParallelIOPipeline(
        loader=loader,
        saver=saver,
        prefetch_size=2,
        num_savers=1,
    )

    results = pipeline.process_batch([first, second], processor)

    assert [result.succeeded for result in results] == [True, True]
    assert events.index("process-second") < events.index("save-first-end")
