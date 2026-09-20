"""Bind managed photographic execution paths to immutable admitted attempts.

Revision ID: 0007_photography_bindings
Revises: 0006_periodic_generation_scrub
"""

import sqlalchemy as sa
from alembic import op

# Alembic exposes these operation methods through its dynamic migration proxy.
# pylint: disable=no-member

revision = "0007_photography_bindings"
down_revision = "0006_periodic_generation_scrub"
branch_labels = None
depends_on = None


def _replace_identity_guard(*, with_bindings: bool) -> None:
    old_bindings = ", OLD.execution_bindings, OLD.execution_bindings_digest" if with_bindings else ""
    new_bindings = ", NEW.execution_bindings, NEW.execution_bindings_digest" if with_bindings else ""
    op.execute(f"""
        CREATE OR REPLACE FUNCTION tp_protect_dispatch_identity()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
        IF ROW(
            OLD.job_id, OLD.attempt_id, OLD.dispatch_id, OLD.tenant_id,
            OLD.plan_digest, OLD.api_version, OLD.output_root,
            OLD.requested_output_root, OLD.admitted_at{old_bindings}
        )
         IS DISTINCT FROM ROW(
            NEW.job_id, NEW.attempt_id, NEW.dispatch_id, NEW.tenant_id,
            NEW.plan_digest, NEW.api_version, NEW.output_root,
            NEW.requested_output_root, NEW.admitted_at{new_bindings}
        )
        THEN
            RAISE EXCEPTION 'immutable dispatch identity';
        END IF;
        IF OLD.state NOT IN ('queued', 'running') AND NOT (
        current_setting('tp.operational_write', true) IS NOT DISTINCT FROM 'on'
        AND (to_jsonb(OLD) - 'generation_id' - 'manifest_digest' - 'cleaned_at' - 'cleanup_checked_at')
        = (to_jsonb(NEW) - 'generation_id' - 'manifest_digest' - 'cleaned_at' - 'cleanup_checked_at')
        AND ((NEW.generation_id IS NOT DISTINCT FROM OLD.generation_id
             AND NEW.manifest_digest IS NOT DISTINCT FROM OLD.manifest_digest)
        OR (NEW.generation_id IS NULL AND NEW.manifest_digest IS NULL)))
        THEN
            RAISE EXCEPTION 'terminal dispatch tombstone';
        END IF;
        RETURN NEW;
        END
        $$
        """)


def upgrade() -> None:
    op.add_column("dispatch_attempts", sa.Column("execution_bindings", sa.LargeBinary(), nullable=True))
    op.add_column("dispatch_attempts", sa.Column("execution_bindings_digest", sa.String(64), nullable=True))
    op.create_check_constraint(
        "ck_dispatch_execution_bindings",
        "dispatch_attempts",
        "(execution_bindings IS NULL AND execution_bindings_digest IS NULL) OR "
        "(execution_bindings IS NOT NULL AND execution_bindings_digest IS NOT NULL AND "
        "octet_length(execution_bindings) BETWEEN 1 AND 65536 AND "
        "execution_bindings_digest ~ '^[0-9a-f]{64}$')",
    )
    _replace_identity_guard(with_bindings=True)


def downgrade() -> None:
    # V5 attempts remain authoritative tombstones after completion. Removing
    # their physical bindings would destroy part of the admitted authority.
    op.execute("LOCK TABLE dispatch_attempts IN ACCESS EXCLUSIVE MODE")
    op.execute("""
        DO $$ BEGIN
        IF EXISTS (
            SELECT 1 FROM dispatch_attempts AS attempt
            JOIN dispatch_plans AS plan ON plan.digest = attempt.plan_digest
            WHERE attempt.execution_bindings IS NOT NULL
               OR attempt.execution_bindings_digest IS NOT NULL
               OR convert_from(plan.canonical_bytes, 'UTF8')::jsonb ->> 'schema' = 'tp.execution.plan.v4'
        ) THEN
            RAISE EXCEPTION 'cannot remove immutable photography execution bindings while V5 attempts exist';
        END IF;
        END $$
        """)
    _replace_identity_guard(with_bindings=False)
    op.drop_constraint("ck_dispatch_execution_bindings", "dispatch_attempts", type_="check")
    op.drop_column("dispatch_attempts", "execution_bindings_digest")
    op.drop_column("dispatch_attempts", "execution_bindings")
