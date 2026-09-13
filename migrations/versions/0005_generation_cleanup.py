"""Track bounded cleanup of immutable attempt workspaces and staging.

Revision ID: 0005_generation_cleanup
Revises: 0004_bounded_event_replay
"""

import sqlalchemy as sa
from alembic import op

# Alembic exposes these operation methods through its dynamic migration proxy.
# pylint: disable=no-member


revision = "0005_generation_cleanup"
down_revision = "0004_bounded_event_replay"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("dispatch_attempts", sa.Column("cleaned_at", sa.Float(), nullable=True))
    op.execute("""
        CREATE OR REPLACE FUNCTION tp_protect_dispatch_identity()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
        IF ROW(
            OLD.job_id, OLD.attempt_id, OLD.dispatch_id, OLD.tenant_id,
            OLD.plan_digest, OLD.api_version, OLD.output_root,
            OLD.requested_output_root, OLD.admitted_at
        )
         IS DISTINCT FROM ROW(
            NEW.job_id, NEW.attempt_id, NEW.dispatch_id, NEW.tenant_id,
            NEW.plan_digest, NEW.api_version, NEW.output_root,
            NEW.requested_output_root, NEW.admitted_at
        )
        THEN
            RAISE EXCEPTION 'immutable dispatch identity';
        END IF;
        IF OLD.state NOT IN ('queued', 'running') AND NOT (
        current_setting('tp.operational_write', true) IS NOT DISTINCT FROM 'on'
        AND (to_jsonb(OLD) - 'generation_id' - 'manifest_digest' - 'cleaned_at')
        = (to_jsonb(NEW) - 'generation_id' - 'manifest_digest' - 'cleaned_at')
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


def downgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION tp_protect_dispatch_identity()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
        IF ROW(
            OLD.job_id, OLD.attempt_id, OLD.dispatch_id, OLD.tenant_id,
            OLD.plan_digest, OLD.api_version, OLD.output_root,
            OLD.requested_output_root, OLD.admitted_at
        )
         IS DISTINCT FROM ROW(
            NEW.job_id, NEW.attempt_id, NEW.dispatch_id, NEW.tenant_id,
            NEW.plan_digest, NEW.api_version, NEW.output_root,
            NEW.requested_output_root, NEW.admitted_at
        )
        THEN
            RAISE EXCEPTION 'immutable dispatch identity';
        END IF;
        IF OLD.state NOT IN ('queued', 'running') AND NOT (current_setting('tp.operational_write', true) IS NOT DISTINCT FROM 'on' AND NEW.generation_id IS NULL AND NEW.manifest_digest IS NULL AND (to_jsonb(OLD) - 'generation_id' - 'manifest_digest')
        = (to_jsonb(NEW) - 'generation_id' - 'manifest_digest')) THEN
            RAISE EXCEPTION 'terminal dispatch tombstone';
        END IF;
        RETURN NEW;
        END
        $$
        """)
    op.drop_column("dispatch_attempts", "cleaned_at")
