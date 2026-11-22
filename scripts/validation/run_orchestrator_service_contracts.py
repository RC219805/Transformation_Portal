#!/usr/bin/env python3
"""Run orchestrator service contracts against explicitly disposable resources."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qsl, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
TESTS = "tests/orchestrator"
PLAIN_FILES = (
    "test_repository_contract.py",
    "test_restart_recovery.py",
    "test_postgres_app_authority.py",
    "test_postgres_event_retention.py",
)
PROVIDER_FILES = (
    "test_dispatch_authority_services.py",
    "test_tenant_dispatch_services.py",
    "test_job_submission_services.py",
)
MIGRATED_FILES = PROVIDER_FILES + (
    "test_upload_receipt_services.py",
    "test_photography_storage_services.py",
    "test_managed_lux_depth_services.py",
    "test_managed_photography_services.py",
    "test_managed_v6_services.py",
)
REVIEW_FILE = "test_photographic_review_store.py"
QUEUE_FILE = "test_queue_contract.py"
DISPATCH_S3 = f"{PROVIDER_FILES[0]}::test_minio_generation_is_conditional_digest_verified_and_revocable"
PAID_PILOT_S3 = "test_paid_pilot_services_contract.py::test_paid_pilot_backend_services_compose_end_to_end"
SERVICE_SELECTION = "(unit or integration) and not ml and not slow and not benchmark"
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
ALEMBIC_FAILURES = {
    "asyncpg.exceptions.InvalidPasswordError": "database authentication",
    "asyncpg.exceptions.InsufficientPrivilegeError": "database privileges",
    "asyncpg.exceptions.InvalidCatalogNameError": "database name",
    "ModuleNotFoundError": "missing Python dependency",
    "TimeoutError": "connection or operation timeout",
    "sqlalchemy.exc.OperationalError": "database connection or operation",
    "sqlalchemy.exc.ProgrammingError": "database statement or schema",
}


class ContractError(RuntimeError):
    """The run could not safely establish live service evidence."""


@dataclass(frozen=True)
class Suite:
    name: str
    args: tuple[str, ...]
    required_tests: tuple[str, ...]
    # Each tuple names a module and, for parametrized modules, its real backend.
    required_groups: tuple[tuple[str, str], ...]
    allowed_skips: tuple[str, ...] = ()


PLAIN = Suite(
    "plain-postgres",
    tuple(f"{TESTS}/{name}" for name in PLAIN_FILES),
    (
        f"{PLAIN_FILES[0]}::test_create_then_get_round_trip[postgres]",
        f"{PLAIN_FILES[1]}::test_sweep_marks_running_and_queued_jobs_worker_lost[postgres]",
        f"{PLAIN_FILES[2]}::test_postgres_legacy_artifact_delete_refuses_an_overlapping_durable_job",
        f"{PLAIN_FILES[2]}::test_postgres_reclaim_retries_an_uncached_job_between_runner_exit_and_finalization[False]",
        f"{PLAIN_FILES[2]}::test_postgres_reclaim_retries_an_uncached_job_between_runner_exit_and_finalization[True]",
        f"{PLAIN_FILES[3]}::test_durable_dispatch_authority_survives_legacy_cleanup_and_restart_sweep",
    ),
    ((PLAIN_FILES[0], "[postgres]"), (PLAIN_FILES[1], "[postgres]"), (PLAIN_FILES[2], ""), (PLAIN_FILES[3], "")),
    (f"{PLAIN_FILES[0]}::test_postgres_delete_job_cascades_events[memory]",),
)
PROVIDER = Suite(
    "provider-authority",
    (*[f"{TESTS}/{name}" for name in PROVIDER_FILES], f"--deselect={TESTS}/{DISPATCH_S3}"),
    (
        f"{PROVIDER_FILES[0]}::test_database_rejects_immutable_identity_and_unfenced_projection_mutation",
        f"{PROVIDER_FILES[1]}::test_signed_tenants_isolate_real_external_worker_artifacts",
        f"{PROVIDER_FILES[2]}::test_process_exit_after_commit_keeps_receipt_and_only_one_dispatch",
    ),
    tuple((name, "") for name in PROVIDER_FILES),
)
MIGRATED = Suite(
    "migrated-postgres-and-redis",
    (
        TESTS,
        *[f"--ignore={TESTS}/{name}" for name in (*PLAIN_FILES, REVIEW_FILE)],
        f"--deselect={TESTS}/{DISPATCH_S3}",
        f"--deselect={TESTS}/{PAID_PILOT_S3}",
    ),
    (
        *PROVIDER.required_tests,
        f"{QUEUE_FILE}::test_reclaim_expired_leases_requeues_abandoned_jobs[redis]",
        "test_upload_receipt_services.py::test_committed_reservation_is_visible_in_a_fresh_python_process",
        "test_upload_receipt_services.py::test_get_recovers_seal_database_gap_but_rejects_other_owner_and_changed_bytes",
    ),
    (*[(name, "") for name in MIGRATED_FILES], (QUEUE_FILE, "[redis]")),
    ("test_artifact_store_contract.py::test_s3_head_downloads_each_stored_version_once[local]",),
)
REVIEW = Suite(
    "migrated-photographic-review",
    (f"{TESTS}/{REVIEW_FILE}",),
    (f"{REVIEW_FILE}::test_postgres_approval_survives_separate_python_process_restart[postgres]",),
    ((REVIEW_FILE, "[postgres]"),),
    tuple(
        f"{REVIEW_FILE}::{name}"
        for name in (
            "test_postgres_corrupt_projection_cannot_be_read_repaired_or_audited[memory-*",
            "test_new_postgres_store_instance_observes_committed_review_and_audit[memory]",
            "test_postgres_native_materials_receipt_and_diagnostic_closure_survive_separate_process_restart[memory]",
            "test_postgres_delivery_recovery_metadata_survives_separate_process_restart[memory]",
            "test_postgres_approval_survives_separate_python_process_restart[memory]",
        )
    ),
)


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str = field(repr=False)
    host: str
    port: int
    database: str

    @property
    def identity(self) -> tuple[str, int, str]:
        host = "loopback" if self.host in LOOPBACK else self.host
        return host, self.port, self.database


def endpoint(name: str, *, redis: bool = False) -> Endpoint:
    url = os.environ.get(name, "").strip()
    try:
        if any(char.isspace() or ord(char) < 32 for char in url):
            raise ValueError
        parsed = urlsplit(url)
        schemes = {"redis", "rediss"} if redis else {"postgresql+asyncpg"}
        host = (parsed.hostname or "").rstrip(".")
        port = parsed.port if parsed.port is not None else (6379 if redis else 5432)
        database = parsed.path.removeprefix("/") or ("0" if redis else "")
        overrides = {key.lower() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}
        if not url or parsed.scheme not in schemes or not host or not port or parsed.netloc.endswith(":") or parsed.fragment:
            raise ValueError
        if not database or "%" in database or "/" in database or any(char.isspace() for char in database):
            raise ValueError
        if overrides & {"host", "port", "database", "dbname", "user", "db", "path"}:
            raise ValueError
        if redis:
            if not database.isascii() or not database.isdigit():
                raise ValueError
            database = str(int(database))
    except ValueError:
        raise ContractError(f"{name} must be an explicit valid service URL with no identity overrides") from None
    return Endpoint(name, url, host, port, database)


async def database_info(db: Endpoint, *, read_revision: bool = False) -> tuple[str, str | None]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(db.url, poolclass=NullPool, connect_args={"timeout": 10, "command_timeout": 10})
    try:
        async with engine.connect() as connection:
            current, version = (
                await connection.execute(text("SELECT current_database(), current_setting('server_version')"))
            ).one()
            if current != db.database:
                raise ContractError(f"{db.name} connected to a database other than its configured target")
            revision = None
            if read_revision:
                revision = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
    finally:
        await engine.dispose()
    return version, revision


async def service_info(db: Endpoint, *, read_revision: bool = False) -> tuple[str, str | None]:
    try:
        return await database_info(db, read_revision=read_revision)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"{db.name} read-only PostgreSQL probe failed ({type(exc).__name__})") from None


async def redis_info(redis: Endpoint, *, ci: bool) -> str:
    from redis.asyncio import Redis

    try:
        async with Redis.from_url(redis.url, socket_connect_timeout=5, socket_timeout=5) as client:
            if not await client.ping():
                raise ContractError(f"{redis.name} Redis PING failed")
            return (await client.info("server"))["redis_version"] if ci else "reachable"
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"{redis.name} Redis probe failed ({type(exc).__name__})") from None


async def preflight(plain: Endpoint, migrated: Endpoint, redis: Endpoint, *, ci: bool) -> dict[str, str]:
    if not migrated.database.endswith("_test") or migrated.database == plain.database:
        raise ContractError("The migrated database must have a distinct *_test name")
    if ci:
        if not plain.database.endswith("_test") or any(value.host not in LOOPBACK for value in (plain, migrated, redis)):
            raise ContractError("CI service contracts require loopback services and a separate plain *_test database")
    else:
        staging = endpoint("TP_DATABASE_URL")
        if staging.database in {plain.database, migrated.database}:
            raise ContractError("Provider test database names must be separate from staging, even across endpoints")
        if any(redis.identity == endpoint(name, redis=True).identity for name in ("TP_REDIS_URL", "TP_FRONTDOOR_REDIS_URL")):
            raise ContractError("TP_TEST_REDIS_URL must select disposable Redis outside staging queue and sessions")
        await service_info(staging)
    plain_version, _ = await service_info(plain)
    migrated_version, _ = await service_info(migrated)
    redis_version = await redis_info(redis, ci=ci)
    if ci and (
        not plain_version.startswith("16.") or not migrated_version.startswith("16.") or not redis_version.startswith("7.")
    ):
        raise ContractError("CI service contracts require PostgreSQL 16 and Redis 7")
    return {"plain_postgres": plain_version, "migrated_postgres": migrated_version, "redis": redis_version}


def test_environment(plain: Endpoint, migrated: Endpoint, redis: Endpoint) -> dict[str, str]:
    pytest_overrides = {"PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"}
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("TP_", "AWS_", "PG")) and key not in pytest_overrides
    }
    env.update(
        PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TP_ORCHESTRATOR_STATE_BACKEND="memory",
        TP_ORCHESTRATOR_QUEUE_BACKEND="memory",
        TP_ARTIFACT_STORE="local",
        TP_TEST_POSTGRES_URL=plain.url,
        TP_TEST_REDIS_URL=redis.url,
        TP_DISPATCH_TEST_REDIS_URL=redis.url,
        TP_DISPATCH_TEST_DATABASE_URL=migrated.url,
        TP_TENANT_TEST_DATABASE_URL=migrated.url,
        TP_SUBMISSION_TEST_DATABASE_URL=migrated.url,
        TP_UPLOAD_RECEIPT_TEST_DATABASE_URL=migrated.url,
        TP_PHOTOGRAPHY_TEST_DATABASE_URL=migrated.url,
    )
    return env


def run_command(args: tuple[str, ...], env: dict[str, str], *, label: str, ci: bool) -> int:
    result = subprocess.run(
        (sys.executable, "-m", *args),
        cwd=ROOT,
        env=env,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if ci:
        output = result.stdout
        urls = {value for value in env.values() if value.startswith(("postgresql+asyncpg://", "redis://", "rediss://"))}
        values = urls | {unquote(urlsplit(url).password or "") for url in urls}
        for value in sorted(values - {""}, key=len, reverse=True):
            output = output.replace(value, "[redacted]")
        print(output, end="", flush=True)
    if result.returncode:
        print(f"{label} exited with status {result.returncode}", file=sys.stderr)
        if not ci and args[0] == "alembic":
            lines = result.stdout.splitlines()
            reported_classes = {line.partition(":")[0] for line in lines}
            for name, reason in ALEMBIC_FAILURES.items():
                if name in reported_classes:
                    print(f"{label}: {reason}", file=sys.stderr)
            if any(line.startswith("FAILED: New upgrade operations detected") for line in lines):
                print(f"{label}: database schema differs from the ORM metadata", file=sys.stderr)
    return result.returncode


def allowed_skip(node: str, suite: Suite) -> bool:
    # The wildcard is limited to the existing payload-version parameter of a
    # named memory-only variant. Required backend evidence never uses it.
    return any(
        node == f"{TESTS}/{allowed}"
        or (allowed.endswith("*") and node.startswith(f"{TESTS}/{allowed[:-1]}") and node.endswith("]"))
        for allowed in suite.allowed_skips
    )


def report_provider_pytest_failure(report: Path, suite: Suite) -> None:
    """Print fixed outcomes with identifiers taken from the selected committed source."""
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    modules: set[str] = set()
    tests: set[str] = set()
    deselected = {arg.removeprefix("--deselect=") for arg in suite.args if arg.startswith("--deselect=")}
    for file, _ in suite.required_groups:
        path = f"{TESTS}/{file}"
        try:
            source = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT, stderr=subprocess.DEVNULL)
            tree = ast.parse(source)
        except (OSError, subprocess.CalledProcessError, ValueError, SyntaxError):
            continue
        modules.add(path)
        for definition in tree.body:
            if (
                isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
                and definition.name.isascii()
                and definition.name.startswith("test_")
            ):
                tests.add(f"{path}::{definition.name}")
    tests.difference_update(deselected)

    diagnostics: set[str] = set()
    for section in ("collectors", "tests"):
        records = payload.get(section)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            node = record.get("nodeid")
            outcome = record.get("outcome")
            if not isinstance(node, str) or outcome not in ("failed", "error", "skipped", "xfailed", "xpassed"):
                continue
            if outcome == "skipped" and allowed_skip(node, suite):
                continue
            if section == "collectors":
                label = node
                phase = "collection"
                if label not in modules:
                    continue
            else:
                # Runtime parameter IDs may contain provider-controlled data.
                label = node.partition("[")[0]
                phase = "test"
                if label not in tests:
                    continue
            for stage in ("setup", "call", "teardown"):
                details = record.get(stage)
                if isinstance(details, dict) and details.get("outcome") in ("failed", "skipped"):
                    phase = stage
                    break
            diagnostics.add(f"{suite.name}: {label} ({phase} {outcome})")
    if not diagnostics:
        print(f"{suite.name}: no verified pytest failure identifiers available", file=sys.stderr)
    for diagnostic in sorted(diagnostics)[:12]:
        print(diagnostic, file=sys.stderr)
    if len(diagnostics) > 12:
        print(f"{suite.name}: {len(diagnostics) - 12} additional pytest diagnostics omitted", file=sys.stderr)


def check_report(report: Path, suite: Suite) -> dict[str, int]:
    """Require actual passing call results in every promised service group."""
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        tests = payload["tests"]
        if payload["exitcode"] != 0 or not isinstance(tests, list) or not tests:
            raise ValueError
        tests_by_id = {case["nodeid"]: case for case in tests}
        counts = Counter(case["outcome"] for case in tests)
        if len(tests_by_id) != len(tests) or not all(isinstance(node, str) and node for node in tests_by_id):
            raise ValueError
        if set(counts) - {"passed", "skipped", "xfailed", "xpassed"}:
            raise ValueError
        passed = {
            node
            for node, case in tests_by_id.items()
            if case["outcome"] == "passed" and case.get("call", {}).get("outcome") == "passed"
        }
        if not passed:
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ContractError(f"{suite.name} did not produce readable, successful pytest results") from None

    for file, backend in suite.required_groups:
        matches = {node for node in tests_by_id if node.startswith(f"{TESTS}/{file}::") and node.endswith(backend)}
        if not matches or matches - passed:
            raise ContractError(f"{suite.name} did not execute and pass every required {file}{backend} service case")
    missing = {f"{TESTS}/{name}" for name in suite.required_tests} - passed
    if missing:
        raise ContractError(f"{suite.name} did not pass required service tests: {', '.join(sorted(missing))}")
    for node in tests_by_id.keys() - passed:
        if tests_by_id[node]["outcome"] != "skipped" or not allowed_skip(node, suite):
            raise ContractError(f"{suite.name} had an unexpected non-passing test")
    return dict(counts)


def prepare_migrated_database(migrated: Endpoint, env: dict[str, str], *, ci: bool) -> str:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = "migrations/alembic.ini"
    migration_env = dict(env, TP_DATABASE_URL=migrated.url)
    for operation in (("upgrade", "head"), ("check",)):
        label = "alembic " + " ".join(operation)
        if run_command(("alembic", "-c", config, *operation), migration_env, label=label, ci=ci):
            raise ContractError(f"{label} failed on the disposable migrated database")
    _, revision = asyncio.run(service_info(migrated, read_revision=True))
    migration_config = Config(str(ROOT / config))
    migration_config.set_main_option("script_location", str(ROOT / "migrations"))
    if not revision or revision != ScriptDirectory.from_config(migration_config).get_current_head():
        raise ContractError("Disposable migrated database did not reach the repository's Alembic head")
    return revision


def run_suite(suite: Suite, env: dict[str, str], directory: Path, *, ci: bool) -> dict[str, object]:
    report = directory / f"{suite.name}.json"
    args = ("pytest", "-qra", "-m", SERVICE_SELECTION, "--json-report", f"--json-report-file={report}", *suite.args)
    print(f"Running {suite.name}", flush=True)
    try:
        if run_command(args, env, label=suite.name, ci=ci):
            raise ContractError(f"{suite.name} pytest run failed")
        counts = check_report(report, suite)
    except ContractError:
        if not ci:
            report_provider_pytest_failure(report, suite)
        raise
    print(f"{suite.name} passed with {counts}", flush=True)
    return {"name": suite.name, "outcomes": counts, "required_tests": suite.required_tests}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ci", "provider-preflight", "provider-plain", "provider"))
    parser.add_argument("--evidence-out", type=Path, help="write a fresh redacted JSON result")
    args = parser.parse_args()
    ci = args.mode == "ci"
    evidence: dict[str, object] = {"mode": args.mode, "status": "failed", "python": sys.version.split()[0]}
    suites: list[dict[str, object]] = []
    evidence["suites"] = suites
    try:
        evidence["tested_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        plain = endpoint("TP_TEST_POSTGRES_URL")
        migrated = endpoint("TP_DISPATCH_TEST_DATABASE_URL")
        redis = endpoint("TP_TEST_REDIS_URL", redis=True)
        evidence["services"] = asyncio.run(preflight(plain, migrated, redis, ci=ci))
        if args.mode != "provider-preflight":
            env = test_environment(plain, migrated, redis)
            if args.mode != "provider-plain":
                evidence["migration"] = prepare_migrated_database(migrated, env, ci=ci)
            with TemporaryDirectory(prefix="tp-orchestrator-service-evidence-") as directory:
                reports = Path(directory)
                if ci:
                    suites.append(run_suite(PLAIN, env, reports, ci=True))
                    suites.append(run_suite(MIGRATED, env, reports, ci=True))
                    # Only photographic review uses the legacy selector on a migrated database.
                    suites.append(run_suite(REVIEW, dict(env, TP_TEST_POSTGRES_URL=migrated.url), reports, ci=True))
                elif args.mode == "provider-plain":
                    suites.append(run_suite(PLAIN, env, reports, ci=False))
                else:
                    suites.append(run_suite(PROVIDER, env, reports, ci=False))
        evidence["status"] = "passed"
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
    if args.evidence_out is not None:
        with args.evidence_out.open("x", encoding="utf-8") as file:
            json.dump(evidence, file, indent=2)
            file.write("\n")
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
