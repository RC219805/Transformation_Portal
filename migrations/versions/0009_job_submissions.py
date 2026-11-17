"""Retain composite-owner submission keys with canonical dispatch tombstones.

Revision ID: 0009_job_submissions
Revises: 0008_photographic_review
"""

import sqlalchemy as sa
from alembic import op

# Alembic exposes these methods through its dynamic migration proxy.
# pylint: disable=no-member

revision = "0009_job_submissions"
down_revision = "0008_photographic_review"
branch_labels = None
depends_on = None


def _identity_guard(with_submissions: bool) -> None:
    fields = (
        "job_id",
        "attempt_id",
        "dispatch_id",
        "tenant_id",
        "plan_digest",
        "api_version",
        "output_root",
        "requested_output_root",
        "admitted_at",
        "execution_bindings",
        "execution_bindings_digest",
    )
    if with_submissions:
        fields += ("submission_key", "submission_owner_sha256", "submission_intent_sha256", "submission_pipeline")
    old, new = (", ".join(f"{prefix}.{field}" for field in fields) for prefix in ("OLD", "NEW"))
    op.execute(f"""
        CREATE OR REPLACE FUNCTION tp_protect_dispatch_identity()
        RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
        IF ROW({old}) IS DISTINCT FROM ROW({new})
        THEN RAISE EXCEPTION 'immutable dispatch identity'; END IF;
        IF OLD.state NOT IN ('queued', 'running') AND NOT (
        current_setting('tp.operational_write', true) IS NOT DISTINCT FROM 'on'
        AND (to_jsonb(OLD) - 'generation_id' - 'manifest_digest' - 'cleaned_at' - 'cleanup_checked_at')
        = (to_jsonb(NEW) - 'generation_id' - 'manifest_digest' - 'cleaned_at' - 'cleanup_checked_at')
        AND ((NEW.generation_id IS NOT DISTINCT FROM OLD.generation_id
             AND NEW.manifest_digest IS NOT DISTINCT FROM OLD.manifest_digest)
        OR (NEW.generation_id IS NULL AND NEW.manifest_digest IS NULL)))
        THEN RAISE EXCEPTION 'terminal dispatch tombstone'; END IF;
        RETURN NEW; END $$
        """)


def upgrade() -> None:
    for field in ("submission_key", "submission_owner_sha256", "submission_intent_sha256", "submission_pipeline"):
        op.add_column("dispatch_attempts", sa.Column(field, sa.String(64), nullable=True))
    op.create_check_constraint(
        "ck_dispatch_submission_identity",
        "dispatch_attempts",
        "(submission_key IS NULL AND submission_owner_sha256 IS NULL AND submission_intent_sha256 IS NULL "
        "AND submission_pipeline IS NULL) OR "
        "(submission_key IS NOT NULL AND submission_owner_sha256 IS NOT NULL AND submission_intent_sha256 IS NOT NULL "
        "AND submission_pipeline IS NOT NULL AND submission_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$' "
        "AND submission_owner_sha256 ~ '^[0-9a-f]{64}$' AND submission_intent_sha256 ~ '^[0-9a-f]{64}$' "
        "AND length(submission_pipeline) BETWEEN 1 AND 64)",
    )
    op.create_index(
        "uq_dispatch_submission_owner_key",
        "dispatch_attempts",
        ["tenant_id", "submission_owner_sha256", "submission_key"],
        unique=True,
        postgresql_where=sa.text("submission_key IS NOT NULL"),
    )
    _identity_guard(True)


def downgrade() -> None:
    op.execute("LOCK TABLE dispatch_attempts IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM dispatch_attempts WHERE submission_key IS NOT NULL)
        THEN RAISE EXCEPTION 'cannot remove consumed submission keys'; END IF;
        END $$""")
    _identity_guard(False)
    op.drop_index("uq_dispatch_submission_owner_key", table_name="dispatch_attempts")
    op.drop_constraint("ck_dispatch_submission_identity", "dispatch_attempts", type_="check")
    for field in ("submission_pipeline", "submission_intent_sha256", "submission_owner_sha256", "submission_key"):
        op.drop_column("dispatch_attempts", field)
