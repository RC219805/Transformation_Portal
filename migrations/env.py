"""Async-aware Alembic environment for the orchestrator schema.

Resolves the database URL from ``TP_DATABASE_URL`` so the same migration
config works for local docker-compose Postgres, CI Postgres service
containers, and production managed Postgres.

Phase 1.B keeps migrations focused on the three orchestrator tables
introduced by ``src/transformation_portal/orchestrator/models.py``.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from transformation_portal.orchestrator.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_database_url() -> str:
    url = os.getenv("TP_DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "TP_DATABASE_URL is not set. Alembic needs a Postgres URL like " "postgresql+asyncpg://user:pw@host:5432/db."
        )
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL without a live engine)."""
    url = _resolve_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async SQLAlchemy engine."""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _resolve_database_url()
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        async with connectable.connect() as connection:
            await connection.run_sync(_do_run_migrations)
    finally:
        await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
