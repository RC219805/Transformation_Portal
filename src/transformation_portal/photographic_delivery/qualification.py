"""Bounded, offline CPU measurements; never a managed grant or human approval."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

SCHEMA = "tp.photo.materials_cpu_qualification.v1"
GIB = 1024**3
MIB = 1024**2
MAX_SOURCE_BYTES = 512 * MIB
MAX_RECIPE_BYTES = 64 * 1024
_MODULES = (
    "transformation_portal.photographic_delivery.qualification",
    "transformation_portal.photographic_delivery.materials",
    "transformation_portal.photographic_delivery.models",
    "transformation_portal.photographic_delivery.native",
    "transformation_portal.core.image_artifact",
    "transformation_portal.core.photographic_export",
    "transformation_portal.ingest.canonical_json",
    "transformation_portal.photographic_delivery.presentation",
    "transformation_portal.orchestrator.photographic_admission",
    "transformation_portal.materials_v4.engine",
    "transformation_portal.materials_v4.contracts",
    "transformation_portal.materials_v4.operations",
    "transformation_portal.lux_depth_v4.photography",
    "transformation_portal.lux_depth_v4.color_preparation",
    "transformation_portal.lux_depth_v5.preview",
    "transformation_portal.lux_depth_v6.color",
)


@dataclass(frozen=True)
class QualificationLimits:
    """An offline ceiling, independent of and unable to enlarge managed limits."""

    memory_bytes: int = 12 * GIB
    scratch_bytes: int = 64 * GIB
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        for name, ceiling in (("memory_bytes", 12 * GIB), ("scratch_bytes", 64 * GIB), ("timeout_seconds", 300)):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError(f"{name} must be a positive integer no greater than {ceiling}")


def _canonical(value: Any) -> bytes:
    from transformation_portal.photographic_delivery.models import canonical_bytes

    return canonical_bytes(value)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are unsupported")
        result[key] = value
    return result


def _read_json(path: Path, maximum: int = MAX_RECIPE_BYTES) -> Any:
    with _open_regular(path, maximum) as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("JSON input exceeds its byte limit")
    return json.loads(raw, object_pairs_hook=_unique)


def _open_regular(path: Path, maximum: int) -> BinaryIO:
    if path != path.resolve(strict=True):
        raise ValueError("Qualification inputs must not traverse symbolic links")
    stream = open(
        path,
        "rb",
        opener=lambda name, flags: os.open(name, flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)),
    )
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
        stream.close()
        raise ValueError("Qualification inputs must be bounded regular files")
    return stream


def _identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def _source(path: Path, expected_sha256: str) -> dict[str, Any]:
    import tifffile

    from transformation_portal.core.image_artifact import validate_source_sha256
    from transformation_portal.lux_depth_v4.photography import validate_input_color_metadata

    validate_source_sha256(expected_sha256)
    with _open_regular(path, MAX_SOURCE_BYTES) as stream:
        original = _identity(os.fstat(stream.fileno()))
        digest = hashlib.sha256()
        while block := stream.read(MIB):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Source bytes differ from the expected SHA-256")
        stream.seek(0)
        color = validate_input_color_metadata(stream, source_name="source.tif", max_pixels=40_000_000)
        stream.seek(0)
        with tifffile.TiffFile(stream) as tif:
            page = tif.pages[0]
            if not isinstance(page, tifffile.TiffPage) or page.dtype is None:
                raise ValueError("Qualification requires a complete photographic TIFF page")
            if page.dtype.kind != "u" or page.dtype.itemsize not in {1, 2}:
                raise ValueError("CPU qualification initially supports integer 8/16-bit SDR TIFF sources")
            width, height = int(page.imagewidth), int(page.imagelength)
            orientation = int(page.tags[274].value) if 274 in page.tags else 1
            if orientation not in range(1, 9):
                raise ValueError("Unsupported TIFF orientation")
            if orientation >= 5:
                width, height = height, width
            bits, alpha = page.dtype.itemsize * 8, page.samplesperpixel in {2, 4}
        if _identity(os.fstat(stream.fileno())) != original:
            raise ValueError("Source changed during header verification")
    return {
        "sha256": expected_sha256,
        "size_bytes": original[2],
        "file_identity": original,
        "width": width,
        "height": height,
        "alpha": alpha,
        "source_bit_depth": bits,
        "orientation": orientation,
        "color_evidence": color,
    }


def _runtime_identity() -> dict[str, Any]:
    modules = {}
    for name in _MODULES:
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise RuntimeError("Qualification requires file-backed implementation modules")
        modules[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    versions: dict[str, str | None] = {
        name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "tifffile", "scipy", "psutil")
    }
    try:
        versions["imagecodecs"] = importlib.metadata.version("imagecodecs")
    except importlib.metadata.PackageNotFoundError:
        versions["imagecodecs"] = None
    with Path(sys.executable).resolve().open("rb") as executable:
        executable_sha256 = hashlib.file_digest(executable, "sha256").hexdigest()
    return {
        "kind": "observed_code_and_package_versions_not_cache_authority",
        "python": platform.python_version(),
        "python_executable_sha256": executable_sha256,
        "platform": {"system": platform.system(), "release": platform.release(), "version": platform.version()},
        "machine": platform.machine(),
        "packages": versions,
        "module_sha256": modules,
    }


def _scratch_usage(root: Path) -> dict[str, int]:
    logical, allocated, seen = 0, 0, set()
    for path in root.rglob("*"):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("Unexpected symbolic link in the private trial namespace")
        if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) not in seen:
            seen.add((info.st_dev, info.st_ino))
            logical += info.st_size
            allocated += info.st_blocks * 512
    return {"logical_bytes": logical, "allocated_bytes": allocated, "accounted_bytes": max(logical, allocated)}


def _write_json(path: Path, value: Any, *, replace_owned_report: bool = False) -> None:
    raw = _canonical(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if replace_owned_report:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Preserve exclusive creation for all new records.
    finally:
        temporary.unlink(missing_ok=True)


def _copy_source(source: Path, destination: Path, bound: dict[str, Any]) -> None:
    digest, count = hashlib.sha256(), 0
    with _open_regular(source, MAX_SOURCE_BYTES) as incoming, destination.open("xb") as outgoing:
        if _identity(os.fstat(incoming.fileno())) != bound["file_identity"]:
            raise ValueError("Retained source identity changed before the trial")
        while block := incoming.read(MIB):
            count += len(block)
            if count > bound["size_bytes"]:
                raise ValueError("Retained source grew during the trial")
            digest.update(block)
            outgoing.write(block)
        if _identity(os.fstat(incoming.fileno())) != bound["file_identity"]:
            raise ValueError("Retained source changed during the trial")
    if count != bound["size_bytes"] or digest.hexdigest() != bound["sha256"]:
        raise ValueError("Retained source snapshot no longer matches its authority")


def _worker(specification: Path) -> None:
    import resource

    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})

    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    spec = _read_json(specification)
    root = specification.parent
    source = Path(spec["source"])
    snapshot = root / "source.snapshot.tif"
    started = time.monotonic()
    import numpy as np

    from transformation_portal.lux_depth_v4.photography import decode_master, linear_to_srgb
    from transformation_portal.photographic_delivery.materials import (
        MaterialsRecipe,
        create_materials_revision,
        validate_materials_revision_receipt,
    )
    from transformation_portal.photographic_delivery.models import DeliveryImage

    if _runtime_identity() != spec["runtime"]:
        raise ValueError("Qualification runtime changed before execution")
    _copy_source(source, snapshot, spec["bound"])
    local = _source(snapshot, spec["bound"]["sha256"])
    if any(local[key] != spec["bound"][key] for key in ("width", "height", "alpha", "source_bit_depth", "orientation")):
        raise ValueError("Immutable snapshot exceeds the admitted source geometry")
    recipe = MaterialsRecipe.from_payload(spec["recipe"])
    image = DeliveryImage("source.tif", local["sha256"], "qualification", source_path=snapshot)
    operation_start = time.monotonic()
    result = create_materials_revision(image, recipe)
    operation_seconds = time.monotonic() - operation_start
    multiplier = 1 if sys.platform == "darwin" else 1024
    operation_return_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)
    receipt = result["receipt"]
    validate_materials_revision_receipt(receipt, receipt["revision_sha256"])
    products = {"master.tif": result["source_bytes"], **{name: item[0] for name, item in result["diagnostic_files"].items()}}
    records = []
    for name, data in products.items():
        with (root / name).open("xb") as stream:
            stream.write(data)
        records.append({"filename": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
    receipt_bytes = _canonical(receipt)
    with (root / "receipt.json").open("xb") as stream:
        stream.write(receipt_bytes)
    records.append(
        {"filename": "receipt.json", "sha256": hashlib.sha256(receipt_bytes).hexdigest(), "size_bytes": len(receipt_bytes)}
    )
    before = image.decode()
    after = decode_master(result["source_bytes"], source_name="master.tif", input_color="auto")
    if before.shape != after.shape or after.source_bit_depth != 16 or (before.alpha is None) != (after.alpha is None):
        raise ValueError("Qualified output geometry, precision or alpha differs")
    before_codes = np.rint(linear_to_srgb(before.pixels) * 65535).astype(np.int32)
    after_codes = np.rint(linear_to_srgb(after.pixels) * 65535).astype(np.int32)
    maximum_delta = int(np.max(np.abs(after_codes - before_codes)))
    pixel_digest = hashlib.sha256(_canonical({"shape": list(after.shape), "alpha": after.alpha is not None}))
    pixel_digest.update(after_codes.astype("<u2").tobytes(order="C"))
    if after.alpha is not None:
        pixel_digest.update(np.rint(after.alpha * 65535).astype("<u2").tobytes(order="C"))
    zero_effect = all(region.strength == 0 for region in recipe.regions)
    if zero_effect and maximum_delta > 1:
        raise ValueError("Zero-effect output changed native pixels beyond one 16-bit code")
    if before.alpha is not None and float(np.max(np.abs(before.alpha - after.alpha))) > 1 / 65535:
        raise ValueError("Native output changed straight alpha")
    verification_complete_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)
    if _runtime_identity() != spec["runtime"]:
        raise ValueError("Qualification runtime changed during execution")
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)
    _write_json(
        root / "result.json",
        {
            "source_sha256": local["sha256"],
            "runtime": spec["runtime"],
            "recipe": spec["recipe"],
            "baseline_rss_bytes": int(baseline * multiplier),
            "peak_rss_bytes": peak,
            "incremental_peak_rss_bytes": max(0, peak - int(baseline * multiplier)),
            "rss_high_water_checkpoints": {
                "semantics": "cumulative_process_high_water_not_phase_exclusive",
                "scope": "offline_worker_not_managed_pipeline",
                "operation_return_bytes": operation_return_peak,
                "independent_verification_complete_bytes": verification_complete_peak,
            },
            "operation_seconds": operation_seconds,
            "worker_seconds": time.monotonic() - started,
            "products": records,
            "zero_effect": zero_effect,
            "maximum_16bit_code_delta": maximum_delta,
            "normalized_pixel_sha256": pixel_digest.hexdigest(),
            "native_changed_pixels": receipt["execution"]["changed_pixels"],
            "approval_transferred": False,
            "photographic_acceptance": "not_established",
        },
    )
    snapshot.unlink()  # Only this fresh trial's owned, verified temporary copy.


@contextmanager
def _supervisor_signals() -> Iterator[None]:
    """Route termination through cleanup; repeated signals cannot interrupt draining."""
    previous = {event: signal.getsignal(event) for event in (signal.SIGINT, signal.SIGTERM)}

    def terminate(event: int, frame: Any) -> None:
        for selected in previous:
            signal.signal(selected, signal.SIG_IGN)
        raise SystemExit(128 + event)

    try:
        for event in previous:
            signal.signal(event, terminate)
        yield
    finally:
        for event, handler in previous.items():
            signal.signal(event, handler)


def _run_trial(
    root: Path,
    spec: dict[str, Any],
    limits: QualificationLimits,
    memory_reservation: int,
    scratch_reservation: int,
    protected_memory: int,
    protected_disk: int,
) -> dict[str, Any]:
    import psutil

    root.mkdir(mode=0o700)
    scratch = root / "scratch"
    scratch.mkdir(mode=0o700)
    _write_json(root / "specification.json", spec)
    environment = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "LANG", "SYSTEMROOT"}}
    environment.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "TMPDIR": str(scratch),
            "TMP": str(scratch),
            "TEMP": str(scratch),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    started, peak, peak_scratch, peak_logical, peak_allocated = time.monotonic(), 0, 0, 0, 0
    minimum_memory, minimum_disk = psutil.virtual_memory().available, shutil.disk_usage(root).free
    with (root / "worker.log").open("xb") as log:
        # Block only the launch/ownership handoff; the child unblocks on entry.
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", _MODULES[0], "_worker", str(root / "specification.json")],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        except BaseException:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            raise
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            owned = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak = max(peak, owned.memory_info().rss)
                    if owned.children(recursive=True):
                        raise RuntimeError("Offline Materials worker unexpectedly created child processes")
                except psutil.NoSuchProcess:
                    pass
                usage = _scratch_usage(root.parent)
                allocated = usage["accounted_bytes"]
                peak_scratch = max(peak_scratch, allocated)
                peak_logical = max(peak_logical, usage["logical_bytes"])
                peak_allocated = max(peak_allocated, usage["allocated_bytes"])
                available, free = psutil.virtual_memory().available, shutil.disk_usage(root).free
                minimum_memory, minimum_disk = min(minimum_memory, available), min(minimum_disk, free)
                if peak > memory_reservation or available < protected_memory:
                    raise RuntimeError("Offline memory or inspection headroom bound exceeded")
                if allocated > scratch_reservation or free < protected_disk + max(0, scratch_reservation - allocated):
                    raise RuntimeError("Offline scratch or protected free-space bound exceeded")
                if (root / "worker.log").stat().st_size > MIB:
                    raise RuntimeError("Offline worker log exceeded its bound")
                if time.monotonic() - started > limits.timeout_seconds:
                    raise TimeoutError("Offline CPU qualification exceeded its wall-time bound")
                time.sleep(0.05)
            if process.returncode:
                raise RuntimeError(f"Offline Materials worker failed; inspect {root.name}/worker.log")
        finally:
            drain_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            try:
                # The leader may already have exited while its owned group survives.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                deadline = time.monotonic() + 3
                while True:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Owned qualification process group did not disappear after SIGKILL")
                    time.sleep(0.01)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, drain_mask)
    result = _read_json(root / "result.json", 256 * 1024)
    if (
        result["runtime"] != spec["runtime"]
        or result["source_sha256"] != spec["bound"]["sha256"]
        or result["recipe"] != spec["recipe"]
    ):
        raise ValueError("Worker evidence does not match its frozen inputs")
    peak = max(peak, result["peak_rss_bytes"])
    usage = _scratch_usage(root.parent)
    peak_scratch = max(peak_scratch, usage["accounted_bytes"])
    peak_logical = max(peak_logical, usage["logical_bytes"])
    peak_allocated = max(peak_allocated, usage["allocated_bytes"])
    if peak > memory_reservation or peak_scratch > scratch_reservation:
        raise RuntimeError("Completed worker exceeded its resource reservation")
    for product in result["products"]:
        path = root / product["filename"]
        if path.name != product["filename"] or path.is_symlink() or path.stat().st_size != product["size_bytes"]:
            raise ValueError("Published qualification product inventory differs")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != product["sha256"]:
                raise ValueError("Published qualification product bytes differ")
    from transformation_portal.photographic_delivery.materials import validate_materials_revision_receipt

    inventory = {entry["filename"]: entry for entry in result["products"]}
    if len(inventory) != len(result["products"]) or set(inventory) != {
        "master.tif",
        "receipt.json",
        "materials-preview.png",
        "materials-overlay.png",
        "materials-diagnostics.json",
    }:
        raise ValueError("Qualification product closure is incomplete")
    receipt = _read_json(root / "receipt.json", 256 * 1024)
    validate_materials_revision_receipt(receipt, inventory["master.tif"]["sha256"])
    if receipt["source_master_sha256"] != spec["bound"]["sha256"] or receipt["recipe"] != spec["recipe"]:
        raise ValueError("Qualification receipt differs from its source or recipe")
    for entry in receipt["diagnostics"]:
        retained = inventory[entry["filename"]]
        if any(retained[key] != entry[key] for key in ("sha256", "size_bytes")):
            raise ValueError("Qualification diagnostics differ from their receipt")
    return {
        **result,
        "observed_peak_rss_bytes": peak,
        "sampled_peak_allocated_bytes": peak_allocated,
        "sampled_peak_logical_bytes": peak_logical,
        "sampled_peak_accounted_bytes": peak_scratch,
        "sampled_minimum_host_available_bytes": minimum_memory,
        "sampled_minimum_disk_free_bytes": minimum_disk,
        "parent_elapsed_seconds": time.monotonic() - started,
    }


def qualify(
    source: Path,
    expected_sha256: str,
    recipe_payload: dict[str, Any],
    output: Path,
    *,
    repeats: int = 3,
    limits: QualificationLimits | None = None,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Measure an explicit offline workload; report managed eligibility separately."""
    import psutil

    from transformation_portal.lux_depth_v5.preview import MAX_PREVIEW_BYTES
    from transformation_portal.orchestrator.photographic_admission import (
        ReviewAdmissionConfig,
        effective_interactive_reserve,
        operation_cost,
    )
    from transformation_portal.photographic_delivery.materials import MaterialsRecipe

    if sys.platform not in {"darwin", "linux"}:
        raise ValueError("Offline CPU qualification initially supports macOS and Linux")
    if type(repeats) is not int or not 1 <= repeats <= 5 or type(preflight_only) is not bool:
        raise ValueError("Qualification requires one to five trials and an explicit planning boolean")
    limits = limits or QualificationLimits()
    source, output = source.expanduser().absolute(), output.expanduser().absolute()
    if output.exists() or output.is_symlink() or output.parent != output.parent.resolve(strict=True):
        raise ValueError("Qualification output must be a fresh directory beneath an existing non-symlink parent")
    if output.is_relative_to(source.parent):
        raise ValueError("Qualification output must remain separate from retained source directories")
    recipe = MaterialsRecipe.from_payload(recipe_payload)
    bound = _source(source, expected_sha256)
    runtime = _runtime_identity()
    config = ReviewAdmissionConfig.from_environment()
    reserve = effective_interactive_reserve(config)
    cost = operation_cost("revision", config, [bound], native_materials_regions=len(recipe.regions))
    product = bound["width"] * bound["height"] * 8 + 65536 + 2 * MAX_PREVIEW_BYTES + 256 * 1024
    scratch = 2 * bound["size_bytes"] + 2 * product * repeats + 16 * MIB
    protected_memory, protected_disk = reserve + 2 * GIB, config.interactive_scratch_bytes + config.scratch_floor_bytes
    available, free = psutil.virtual_memory().available, shutil.disk_usage(output.parent).free
    reasons = []
    if cost.memory_bytes > limits.memory_bytes or scratch > limits.scratch_bytes:
        reasons.append("offline_operation_limit")
    if available < cost.memory_bytes + protected_memory:
        reasons.append("physical_memory_headroom")
    if free < scratch + protected_disk:
        reasons.append("physical_scratch_headroom")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "blocked" if reasons else "planned",
        "blocking_reasons": reasons,
        "scope": "offline_cpu_compositor_only",
        "execution_context": "fresh_process_per_trial_os_cache_uncontrolled",
        "source": {key: value for key, value in bound.items() if key != "file_identity"},
        "recipe": recipe.to_payload(),
        "runtime": runtime,
        "limits": asdict(limits),
        "offline_memory_reservation_bytes": cost.memory_bytes,
        "offline_scratch_reservation_bytes": scratch,
        "protected_memory_headroom_bytes": protected_memory,
        "protected_disk_headroom_bytes": protected_disk,
        "host_available_before_bytes": available,
        "disk_free_before_bytes": free,
        "trials": [],
        "managed_profile": {
            "origin": "command_environment_not_live_server",
            "config": asdict(config),
            "effective_interactive_reserve_bytes": reserve,
            "effective_bulk_memory_bytes": min(config.bulk_memory_bytes, config.memory_bytes - reserve),
            "memory_eligible": cost.memory_bytes <= min(config.bulk_memory_bytes, config.memory_bytes - reserve),
            "live_server_configuration_checked": False,
            "managed_grant": False,
        },
        "managed_eligibility": "not_established_offline_measurement_only",
        "production_flags_changed": False,
        "model_inference_performed": False,
        "photographic_acceptance": "not_established",
        "source_unchanged": False,
    }
    output.mkdir(mode=0o700)
    with _supervisor_signals():
        try:
            if not reasons and not preflight_only:
                for number in range(1, repeats + 1):
                    spec = {"source": str(source), "bound": bound, "recipe": recipe.to_payload(), "runtime": runtime}
                    report["trials"].append(
                        _run_trial(
                            output / f"trial-{number}",
                            spec,
                            limits,
                            cost.memory_bytes,
                            scratch,
                            protected_memory,
                            protected_disk,
                        )
                    )
                report["status"] = "measured"
                report["normalized_pixels_repeatable"] = (
                    len({trial["normalized_pixel_sha256"] for trial in report["trials"]}) == 1
                    if len(report["trials"]) >= 2
                    else None
                )
                if report["normalized_pixels_repeatable"] is False:
                    report["status"] = "not_qualified"
                    report["blocking_reasons"].append("normalized_output_pixels_differ_between_trials")
                measured = max(trial["incremental_peak_rss_bytes"] for trial in report["trials"])
                report["observed_incremental_peak_plus_50_percent_bytes"] = math.ceil(measured * 1.5)
                report["analytical_reservation_covers_measured_margin"] = cost.memory_bytes >= math.ceil(measured * 1.5)
                if not report["analytical_reservation_covers_measured_margin"]:
                    report["status"] = "not_qualified"
                    report["blocking_reasons"].append("measured_margin_exceeds_analytical_reservation")
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            try:
                report["source_unchanged"] = _source(source, expected_sha256) == bound
                if not report["source_unchanged"]:
                    report["status"] = "failed"
            except BaseException as error:
                report["status"] = "failed"
                report["source_revalidation_error"] = {"type": type(error).__name__, "message": str(error)}
                if not isinstance(error, (OSError, ValueError)):
                    raise
            finally:
                try:
                    _write_json(output / "qualification.json", report)
                except (SystemExit, KeyboardInterrupt) as error:
                    report["status"] = "failed"
                    report["report_publication_error"] = {"type": type(error).__name__, "message": str(error)}
                    # This path belongs exclusively to the newly created qualification namespace.
                    _write_json(output / "qualification.json", report, replace_owned_report=True)
                    raise
    if not report["source_unchanged"]:
        raise ValueError("Retained source changed during qualification")
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) == 2 and arguments[0] == "_worker":
        _worker(Path(arguments[1]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Caller-authorized retained 8/16-bit SDR TIFF")
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh private directory outside the source directory")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(arguments)
    try:
        report = qualify(
            args.source,
            args.source_sha256,
            _read_json(args.recipe.absolute()),
            args.output,
            repeats=args.repeats,
            preflight_only=args.preflight_only,
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Qualification failed: {error}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {args.output / 'qualification.json'}")
    return 2 if report["status"] in {"blocked", "not_qualified"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
