"""Private, paired Lux candidate evaluation; never an automatic photographic verdict.

The runner callback is trusted local instrumentation, not an execution-plan loader.
Its receipt binds actual pipeline artifacts to the frozen corpus and identities.
One callback invocation is one independent complete-batch observation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence, TypedDict

from transformation_portal.ingest.canonical_json import dump_json, dumps_json

CORPUS_SCHEMA = "tp.lux.evaluation.corpus.v1"
SPEC_SCHEMA = "tp.lux.evaluation.run_spec.v1"
RECEIPT_SCHEMA = "tp.lux.evaluation.receipt.v1"
RUN_SCHEMA = "tp.lux.evaluation.run.v1"
MIN_REPEATS = 20
SCENARIOS = frozenset({"cold", "warm", "cache_hit", "cache_miss"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".cr2", ".nef", ".arw"})
IDENTITY_DIGESTS = (
    "source_sha256",
    "plan_sha256",
    "runtime_sha256",
    "interpreter_sha256",
    "dependency_sha256",
    "model_sha256",
    "processing_sha256",
)


class EvaluationError(ValueError):
    """Incomplete, unsupported, or changed evidence cannot authorize comparison."""


class _BatchStatistics(TypedDict):
    independent_batches: int
    p50_seconds: float
    p95_seconds: float
    standard_deviation_seconds: float
    raw_seconds: list[float]


def file_sha256(path: Path) -> str:
    """Hash bytes and reject detectable mutation while reading."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise EvaluationError(f"File changed during hashing: {path}")
    return digest.hexdigest()


def _digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) or value == "0" * 64:
        raise EvaluationError(f"{label} must be a non-placeholder lowercase SHA-256 digest")


def _inside(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise EvaluationError("Paths must be nonempty relative paths without parent traversal")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise EvaluationError("Path escapes its evidence/input root")
    return result


def _private_destination(path: Path) -> None:
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists():
            raise EvaluationError("Private evaluation artifacts must be outside a Git worktree")


def _write_private(path: Path, payload: Mapping[str, Any]) -> None:
    _private_destination(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        dump_json(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Cannot read JSON evidence: {path}") from exc
    if not isinstance(result, dict):
        raise EvaluationError(f"Expected JSON object: {path}")
    return result


def _image_metadata(path: Path) -> dict[str, Any]:
    """Read headers only; RAW decoding remains an optional separate acceptance lane."""
    metadata: dict[str, Any] = {"format": path.suffix.lower().lstrip(".")}
    try:
        if path.suffix.lower() in {".tif", ".tiff", ".dng"}:
            import tifffile

            with tifffile.TiffFile(path) as image:
                page = image.pages[0]
                if not isinstance(page, tifffile.TiffPage):
                    raise ValueError("Image metadata requires a complete TIFF page")
                metadata.update(
                    width=int(page.imagewidth),
                    height=int(page.imagelength),
                    dtype=str(page.dtype),
                    samples_per_pixel=int(page.samplesperpixel),
                )
        else:
            from PIL import Image

            with Image.open(path) as image:
                metadata.update(width=image.width, height=image.height, mode=image.mode)
    except (OSError, ValueError, ImportError):
        metadata["inspection"] = "unsupported_without_optional_raw_decoder"
    return metadata


def freeze_corpus(source_root: Path, roots: Sequence[str], output: Path, baseline_commit: str) -> dict[str, Any]:
    """Freeze private file identities without copying or transforming photographic inputs.

    Same-stem RAW/TIFF names form candidate scene groups, requiring human review.
    This grouping never increases the number of independent timing observations.
    """
    source_root = source_root.resolve(strict=True)
    if not isinstance(baseline_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", baseline_commit):
        raise EvaluationError("baseline_commit must be the full Git commit")
    if not roots or len(set(roots)) != len(roots):
        raise EvaluationError("Specify distinct corpus roots")
    files: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for relative_root in roots:
        directory = _inside(source_root, relative_root)
        if not directory.is_dir():
            raise EvaluationError(f"Missing corpus directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(source_root):
                raise EvaluationError("Corpus inputs must be contained regular files, not symlinks")
            if path in paths:
                raise EvaluationError("Corpus roots overlap")
            paths.add(path)
            files.append(
                {
                    "path": str(path.relative_to(source_root)),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                    "scene_id": hashlib.sha256(path.stem.casefold().encode()).hexdigest()[:16],
                    "metadata": _image_metadata(path),
                }
            )
    if not files:
        raise EvaluationError("Corpus contains no supported images")
    files.sort(key=lambda row: row["path"])
    seen: dict[str, str] = {}
    for row in files:
        row["duplicate_of"] = seen.get(row["sha256"])
        seen.setdefault(row["sha256"], row["path"])
    manifest = {
        "schema": CORPUS_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": baseline_commit,
        "source_root": str(source_root),
        "roots": list(roots),
        "files": files,
    }
    _write_private(output, manifest)
    return manifest


def verify_corpus(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Rehash every frozen input, including aliases excluded by exact-byte deduplication."""
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise EvaluationError("Frozen corpus manifest changed")
    manifest = _read_object(path)
    if manifest.get("schema") != CORPUS_SCHEMA or not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise EvaluationError("Invalid or empty corpus manifest")
    root_value = manifest.get("source_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise EvaluationError("Corpus source_root must be absolute")
    root = Path(root_value).resolve(strict=True)
    seen: dict[str, str] = {}
    paths: set[str] = set()
    for row in manifest["files"]:
        if not isinstance(row, dict):
            raise EvaluationError("Invalid corpus file record")
        source = _inside(root, row.get("path"))
        if row["path"] in paths or (root / row["path"]).is_symlink():
            raise EvaluationError("Duplicate path or symlink in corpus")
        paths.add(row["path"])
        _digest(row.get("sha256"), "input sha256")
        if source.stat().st_size != row.get("size_bytes") or file_sha256(source) != row["sha256"]:
            raise EvaluationError(f"Frozen input changed: {row['path']}")
        if row.get("duplicate_of") != seen.get(row["sha256"]):
            raise EvaluationError("Corpus exact-byte deduplication changed")
        seen.setdefault(row["sha256"], row["path"])
    return manifest


def validate_spec(spec: Mapping[str, Any]) -> None:
    """Keep algorithm, model, and infrastructure experiments independently attributable."""
    if not isinstance(spec, Mapping) or spec.get("schema") != SPEC_SCHEMA:
        raise EvaluationError("Unsupported run-spec schema")
    _digest(spec.get("corpus_sha256"), "corpus_sha256")
    repeats = spec.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < MIN_REPEATS:
        raise EvaluationError(f"Candidate comparison requires at least {MIN_REPEATS} independent pairs")
    scenarios = spec.get("scenarios")
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or any(not isinstance(item, str) or item not in SCENARIOS for item in scenarios)
        or len(set(scenarios)) != len(scenarios)
    ):
        raise EvaluationError("Specify distinct supported scenarios")
    experiment = spec.get("experiment")
    if experiment not in {"infrastructure", "performance", "model", "photography"}:
        raise EvaluationError("Unknown experiment; do not mix model, photography, and infrastructure changes")
    variants = spec.get("variants")
    if not isinstance(variants, dict) or set(variants) != {"v3", "v4"}:
        raise EvaluationError("Both v3 and v4 variants are required")
    for variant in variants.values():
        if not isinstance(variant, dict) or not isinstance(variant.get("identity"), dict):
            raise EvaluationError("Variant identity is required")
        identity = variant["identity"]
        if set(identity) != {"implementation_commit", *IDENTITY_DIGESTS}:
            raise EvaluationError("Variant identity must contain exactly the documented identity fields")
        commit = identity.get("implementation_commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise EvaluationError("implementation_commit must be a full Git commit")
        for key in IDENTITY_DIGESTS:
            _digest(identity.get(key), key)
    shared = {"interpreter_sha256", "dependency_sha256"}
    if experiment != "model":
        shared.add("model_sha256")
    if experiment != "photography":
        shared.add("processing_sha256")
    for key in shared:
        if variants["v3"]["identity"][key] != variants["v4"]["identity"][key]:
            raise EvaluationError(f"{key} differs across this {experiment} experiment")


@dataclass(frozen=True)
class EvaluationRequest:
    """Trusted instrumentation request; runners must complete publication before returning."""

    variant: str
    scenario: str
    pair_index: int
    output_dir: Path
    corpus_path: Path
    corpus_sha256: str
    identity: Mapping[str, Any]
    host: Mapping[str, str]


Runner = Callable[[EvaluationRequest], Mapping[str, Any]]


def _host() -> dict[str, str]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "node": platform.node(),
        "platform": platform.platform(),
    }


def _sequence(spec: Mapping[str, Any]) -> Iterator[tuple[str, int, str]]:
    for scenario in spec["scenarios"]:
        for pair in range(spec["repeats"]):
            for variant in (("v3", "v4") if pair % 2 == 0 else ("v4", "v3")):
                yield scenario, pair, variant


def _request(
    spec: Mapping[str, Any],
    corpus_path: Path,
    root: Path,
    scenario: str,
    pair: int,
    variant: str,
    host: Mapping[str, str] | None = None,
) -> EvaluationRequest:
    return EvaluationRequest(
        variant,
        scenario,
        pair,
        root / scenario / f"pair_{pair:04d}" / variant,
        corpus_path,
        spec["corpus_sha256"],
        MappingProxyType(dict(spec["variants"][variant]["identity"])),
        MappingProxyType(dict(_host() if host is None else host)),
    )


def _validate_receipt(receipt: Mapping[str, Any], request: EvaluationRequest, input_hashes: set[str]) -> None:
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("complete") is not True:
        raise EvaluationError("Missing complete evaluation receipt")
    backend = receipt.get("executed_backend")
    if (
        receipt.get("synthetic") is not False
        or not isinstance(backend, str)
        or backend.strip().lower() in {"", "synthetic", "stub"}
    ):
        raise EvaluationError("Synthetic or unspecified execution is not candidate evidence")
    if receipt.get("corpus_sha256") != request.corpus_sha256 or receipt.get("identity") != request.identity:
        raise EvaluationError("Execution identity differs from the frozen experiment")
    if receipt.get("host") != request.host:
        raise EvaluationError("Execution host differs from the paired evaluation host")
    inputs = receipt.get("input_sha256s")
    if not isinstance(inputs, list) or any(not isinstance(item, str) for item in inputs):
        raise EvaluationError("Missing input execution identities")
    if len(inputs) != len(set(inputs)) or set(inputs) != input_hashes:
        raise EvaluationError("Receipt did not execute every unique frozen input exactly once")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EvaluationError("Missing published artifacts")
    image_inputs: set[str] = set()
    artifact_paths: set[Path] = set()
    evidence_found = False
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise EvaluationError("Invalid artifact record")
        path = _inside(request.output_dir, artifact.get("path"))
        if path in artifact_paths or (request.output_dir / artifact["path"]).is_symlink():
            raise EvaluationError("Duplicate artifact path or symlink")
        artifact_paths.add(path)
        _digest(artifact.get("sha256"), "artifact sha256")
        if not path.is_file() or file_sha256(path) != artifact["sha256"]:
            raise EvaluationError("Published artifact missing or changed")
        if artifact.get("kind") == "image":
            if artifact.get("input_sha256") not in input_hashes:
                raise EvaluationError("Image artifact is not bound to the corpus")
            image_inputs.add(artifact["input_sha256"])
        elif artifact.get("kind") == "evidence":
            if not _read_object(path).get("schema"):
                raise EvaluationError("Execution evidence must carry a schema")
            evidence_found = True
        else:
            raise EvaluationError("Unsupported artifact kind")
    if image_inputs != input_hashes or not evidence_found:
        raise EvaluationError("Every input needs an output image and the batch needs execution evidence")
    if request.scenario == "warm":
        if (
            receipt.get("warmup_complete") is not True
            or not isinstance(receipt.get("session_id"), str)
            or not receipt["session_id"]
        ):
            raise EvaluationError("Warm evaluation requires an initialized persistent backend session")
    if request.scenario.startswith("cache_"):
        if receipt.get("cache_authorized") is not True or receipt.get("cache_state") != request.scenario.removeprefix(
            "cache_"
        ):
            raise EvaluationError("Cache scenario lacks matching governed cache evidence")


def command_runner(spec: Mapping[str, Any]) -> Runner:
    """Build a shell-free cold subprocess runner. Stateful warm/cache runners use the Python API."""
    if spec.get("scenarios") != ["cold"]:
        raise EvaluationError("Command runner supports cold only; warm/cache need an instrumented stateful Python callback")
    timeout = spec.get("timeout_seconds", 3600)
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise EvaluationError("timeout_seconds must be finite and positive")
    commands = {}
    for variant in ("v3", "v4"):
        command = spec["variants"][variant].get("command")
        if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise EvaluationError("Each cold variant requires an explicit argv command")
        commands[variant] = list(command)

    def execute(request: EvaluationRequest) -> Mapping[str, Any]:
        receipt_path = request.output_dir / "receipt.json"
        env = {
            **os.environ,
            "TP_LUX_EVAL_RECEIPT": str(receipt_path),
            "TP_LUX_EVAL_OUTPUT_DIR": str(request.output_dir),
            "TP_LUX_EVAL_CORPUS": str(request.corpus_path),
            "TP_LUX_EVAL_VARIANT": request.variant,
            "TP_LUX_EVAL_SCENARIO": request.scenario,
            "TP_LUX_EVAL_PAIR_INDEX": str(request.pair_index),
        }
        with (request.output_dir / "stdout.log").open("x") as stdout, (request.output_dir / "stderr.log").open("x") as stderr:
            try:
                with subprocess.Popen(
                    commands[request.variant], env=env, stdout=stdout, stderr=stderr, start_new_session=True
                ) as process:
                    try:
                        returncode = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        # The wrapper may own an inference subprocess. Bound the
                        # whole private process group, not just its immediate parent.
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait()
                        raise
                    if returncode != 0:
                        raise subprocess.CalledProcessError(returncode, commands[request.variant])
            except (subprocess.SubprocessError, OSError) as exc:
                raise EvaluationError("Candidate command failed; partial observations are not acceptance") from exc
        return _read_object(receipt_path)

    return execute


def run_evaluation(corpus_path: Path, spec: Mapping[str, Any], output_dir: Path, runner: Runner) -> Path:
    """Measure independently repeated V3/V4 batches and persist each validated observation.

    Warm callbacks must own actual initialized sessions outside this timing boundary.
    Callback duration covers batch completion/publication; cold command duration also
    includes preparation, process startup, backend initialization and receipt writing.
    """
    spec = json.loads(dumps_json(spec, allow_nan=False))
    validate_spec(spec)
    corpus_path = corpus_path.resolve(strict=True)
    corpus = verify_corpus(corpus_path, spec["corpus_sha256"])
    _private_destination(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    output_dir = output_dir.resolve()
    _write_private(output_dir / "spec.json", spec)
    input_hashes = {row["sha256"] for row in corpus["files"]}
    observations = []
    sessions: dict[str, str] = {}
    host = _host()
    for scenario, pair, variant in _sequence(spec):
        verify_corpus(corpus_path, spec["corpus_sha256"])
        request = _request(spec, corpus_path, output_dir, scenario, pair, variant, host)
        request.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        start = time.perf_counter()
        receipt = dict(runner(request))
        elapsed = time.perf_counter() - start
        _validate_receipt(receipt, request, input_hashes)
        verify_corpus(corpus_path, spec["corpus_sha256"])
        if scenario == "warm":
            session = receipt["session_id"]
            if variant in sessions and sessions[variant] != session:
                raise EvaluationError("Warm backend session changed between observations")
            sessions[variant] = session
        observation = {
            "variant": variant,
            "scenario": scenario,
            "pair_index": pair,
            "elapsed_seconds": elapsed,
            "receipt": receipt,
        }
        _write_private(request.output_dir / "observation.json", observation)
        observations.append(observation)
    run_path = output_dir / "run.json"
    _write_private(
        run_path,
        {
            "schema": RUN_SCHEMA,
            "host": host,
            "corpus_path": str(corpus_path),
            "spec_sha256": file_sha256(output_dir / "spec.json"),
            "observations": observations,
        },
    )
    return run_path


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _paired_uncertainty(variants: Mapping[str, list[float]], *, seed: str) -> dict[str, Any]:
    """Resample independent batch pairs together, without inventing image samples.

    This descriptive percentile-bootstrap interval is not the production policy's
    confidence gate. In particular, twenty pairs poorly resolve tail latency.
    """
    baseline, candidate = variants["v3"], variants["v4"]
    if len(baseline) != len(candidate) or len(baseline) < MIN_REPEATS:
        raise EvaluationError("Uncertainty requires complete independent batch pairs")
    generator = random.Random(seed)
    draws = 10000
    deltas = []
    for _ in range(draws):
        indices = [generator.randrange(len(baseline)) for _ in baseline]
        reference = _percentile([baseline[index] for index in indices], 0.95)
        measured = _percentile([candidate[index] for index in indices], 0.95)
        deltas.append((measured / reference - 1) * 100)
    return {
        "method": "paired_percentile_bootstrap",
        "independent_pairs": len(baseline),
        "resamples": draws,
        "confidence_level": 0.95,
        "p95_delta_percent_interval": [_percentile(deltas, 0.025), _percentile(deltas, 0.975)],
        "seed_sha256": seed,
        "limitations": "Descriptive finite-sample interval; tail latency and host drift remain uncertain",
        "production_confidence_gate": "not_evaluated",
    }


def compare_run(run_path: Path) -> dict[str, Any]:
    """Revalidate complete local evidence; photographic acceptance always remains separate."""
    run = _read_object(run_path)
    if run.get("schema") != RUN_SCHEMA:
        raise EvaluationError("Unsupported run schema")
    root = run_path.resolve().parent
    if file_sha256(root / "spec.json") != run.get("spec_sha256"):
        raise EvaluationError("Frozen experiment specification changed")
    spec = _read_object(root / "spec.json")
    validate_spec(spec)
    host = run.get("host")
    if not isinstance(host, dict) or set(host) != {"system", "machine", "node", "platform"}:
        raise EvaluationError("Missing paired host identity")
    if any(not isinstance(value, str) or not value for value in host.values()):
        raise EvaluationError("Invalid paired host identity")
    corpus_path = Path(run.get("corpus_path", ""))
    corpus = verify_corpus(corpus_path, spec["corpus_sha256"])
    input_hashes = {row["sha256"] for row in corpus["files"]}
    observations = run.get("observations")
    expected = list(_sequence(spec))
    if not isinstance(observations, list) or len(observations) != len(expected):
        raise EvaluationError("Incomplete independent paired observations")
    samples: dict[str, dict[str, list[float]]] = {scenario: {"v3": [], "v4": []} for scenario in spec["scenarios"]}
    sessions: dict[str, str] = {}
    for observation, (scenario, pair, variant) in zip(observations, expected):
        if not isinstance(observation, dict) or (
            observation.get("scenario"),
            observation.get("pair_index"),
            observation.get("variant"),
        ) != (scenario, pair, variant):
            raise EvaluationError("Duplicate, missing, or reordered paired observation")
        request = _request(spec, corpus_path, root, scenario, pair, variant, host)
        if _read_object(request.output_dir / "observation.json") != observation:
            raise EvaluationError("Raw observation differs from run summary")
        receipt = observation.get("receipt")
        if not isinstance(receipt, dict):
            raise EvaluationError("Missing observation receipt")
        _validate_receipt(receipt, request, input_hashes)
        if scenario == "warm":
            if variant in sessions and sessions[variant] != receipt["session_id"]:
                raise EvaluationError("Warm backend session changed")
            sessions[variant] = receipt["session_id"]
        elapsed = observation.get("elapsed_seconds")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise EvaluationError("Invalid batch duration")
        samples[scenario][variant].append(elapsed)
    results = {}
    for scenario, variants in samples.items():
        statistics_by_variant: dict[str, _BatchStatistics] = {
            variant: {
                "independent_batches": len(values),
                "p50_seconds": _percentile(values, 0.5),
                "p95_seconds": _percentile(values, 0.95),
                "standard_deviation_seconds": statistics.stdev(values),
                "raw_seconds": values,
            }
            for variant, values in variants.items()
        }
        delta = (statistics_by_variant["v4"]["p95_seconds"] / statistics_by_variant["v3"]["p95_seconds"] - 1) * 100
        results[scenario] = {
            **statistics_by_variant,
            "p95_delta_percent": delta,
            "performance_verdict": "pass" if delta <= 10 else "warn" if delta <= 15 else "fail",
            "uncertainty": _paired_uncertainty(
                variants, seed=hashlib.sha256(f"{file_sha256(run_path)}:{scenario}".encode()).hexdigest()
            ),
        }
    return {
        "schema": "tp.lux.evaluation.comparison.v1",
        "run_sha256": file_sha256(run_path),
        "corpus_sha256": spec["corpus_sha256"],
        "host": run.get("host"),
        "scenarios": results,
        "photographic_review": "required",
        "candidate_acceptance": "not_established",
        "automatic_enforcement_eligible": False,
        "confidence": "descriptive paired bootstrap only; production confidence gate not evaluated",
    }
