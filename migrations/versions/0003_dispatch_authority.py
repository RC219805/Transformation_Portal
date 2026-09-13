"""Atomic admission, immutable dispatch records, and publication fencing.

Revision ID: 0003_dispatch_authority
Revises: 0002_operational_audit_events
"""

# Alembic exposes these operation methods through its dynamic migration proxy.
# pylint: disable=no-member

from alembic import op

revision = "0003_dispatch_authority"
down_revision = "0002_operational_audit_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SEQUENCE dispatch_lease_epoch_seq")
    op.execute("""
        CREATE TABLE admission_capacity (
            scope VARCHAR(160) NOT NULL,
            "limit" INTEGER NOT NULL,
            active INTEGER NOT NULL,
            PRIMARY KEY (scope),
            CHECK ("limit" > 0 AND active >= 0 AND active <= "limit")
        )
        """)
    op.execute("""
        CREATE TABLE dispatch_plans (
            digest VARCHAR(64) NOT NULL,
            canonical_bytes BYTEA NOT NULL,
            PRIMARY KEY (digest),
            CHECK (octet_length(canonical_bytes) <= 1048576)
        )
        """)
    op.execute("""
        CREATE TABLE dispatch_attempts (
            job_id VARCHAR(64) NOT NULL,
            attempt_id VARCHAR(64) NOT NULL,
            dispatch_id VARCHAR(64) NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            plan_digest VARCHAR(64) NOT NULL,
            api_version VARCHAR(16) NOT NULL,
            output_root TEXT NOT NULL,
            requested_output_root TEXT NOT NULL,
            state VARCHAR(32) NOT NULL,
            holder VARCHAR(128),
            lease_epoch BIGINT NOT NULL,
            lease_valid_until FLOAT,
            generation_id VARCHAR(64),
            manifest_digest VARCHAR(64),
            admitted_at FLOAT NOT NULL,
            finished_at FLOAT,
            PRIMARY KEY (job_id),
            CHECK (lease_epoch >= 0),
            UNIQUE (attempt_id),
            UNIQUE (dispatch_id),
            FOREIGN KEY(plan_digest) REFERENCES dispatch_plans (digest)
        )
        """)
    op.execute("CREATE INDEX ix_dispatch_attempts_state ON dispatch_attempts (state)")
    op.execute("CREATE INDEX ix_dispatch_attempts_tenant_id ON dispatch_attempts (tenant_id)")
    op.execute("""
        CREATE TABLE operational_records (
            id BIGSERIAL NOT NULL,
            job_id VARCHAR(64) NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            kind VARCHAR(32) NOT NULL,
            created_at FLOAT NOT NULL,
            canonical_bytes BYTEA NOT NULL,
            digest VARCHAR(64) NOT NULL,
            PRIMARY KEY (id)
        )
        """)
    op.execute("CREATE INDEX ix_operational_records_job_id ON operational_records (job_id)")
    op.execute("""
        CREATE TABLE operational_outbox (
            id BIGSERIAL NOT NULL,
            job_id VARCHAR(64) NOT NULL,
            kind VARCHAR(32) NOT NULL,
            payload JSONB NOT NULL,
            created_at FLOAT NOT NULL,
            delivered_at FLOAT,
            PRIMARY KEY (id)
        )
        """)
    op.execute("CREATE INDEX ix_operational_outbox_job_id ON operational_outbox (job_id)")
    op.execute("CREATE INDEX ix_operational_outbox_delivered_at ON operational_outbox (delivered_at)")
    op.execute("""
        CREATE TABLE committed_generations (
            generation_id VARCHAR(64) NOT NULL,
            job_id VARCHAR(64) NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            manifest_digest VARCHAR(64) NOT NULL,
            manifest_bytes BYTEA NOT NULL,
            created_at FLOAT NOT NULL,
            PRIMARY KEY (generation_id)
        )
        """)
    op.execute("CREATE INDEX ix_committed_generations_job_id ON committed_generations (job_id)")
    op.execute(
        "CREATE FUNCTION tp_reject_evidence_mutation() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'immutable operational evidence'; END $$"
    )
    op.execute("""
        CREATE FUNCTION tp_protect_dispatch_identity() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
        IF ROW(OLD.job_id, OLD.attempt_id, OLD.dispatch_id, OLD.tenant_id, OLD.plan_digest, OLD.api_version, OLD.output_root, OLD.requested_output_root, OLD.admitted_at)
         IS DISTINCT FROM ROW(NEW.job_id, NEW.attempt_id, NEW.dispatch_id, NEW.tenant_id, NEW.plan_digest, NEW.api_version, NEW.output_root, NEW.requested_output_root, NEW.admitted_at)
        THEN RAISE EXCEPTION 'immutable dispatch identity'; END IF;
        IF OLD.state NOT IN ('queued', 'running') AND NOT (current_setting('tp.operational_write', true) IS NOT DISTINCT FROM 'on' AND NEW.generation_id IS NULL AND NEW.manifest_digest IS NULL AND (to_jsonb(OLD) - 'generation_id' - 'manifest_digest') = (to_jsonb(NEW) - 'generation_id' - 'manifest_digest')) THEN RAISE EXCEPTION 'terminal dispatch tombstone'; END IF;
        RETURN NEW; END $$
        """)
    op.execute(
        "CREATE TRIGGER protect_dispatch_identity BEFORE UPDATE ON dispatch_attempts FOR EACH ROW EXECUTE FUNCTION tp_protect_dispatch_identity()"
    )
    op.execute(
        "CREATE TRIGGER protect_dispatch_tombstone BEFORE DELETE ON dispatch_attempts FOR EACH ROW EXECUTE FUNCTION tp_reject_evidence_mutation()"
    )
    op.execute("""
        CREATE FUNCTION tp_protect_job_authority() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
        IF current_setting('tp.operational_write', true) IS DISTINCT FROM 'on'
         AND EXISTS (SELECT 1 FROM dispatch_attempts WHERE job_id = OLD.id)
         AND ROW(OLD.state, OLD.started_at, OLD.finished_at, OLD.exit_code, OLD.cancel_requested, OLD.artifacts, OLD.run_summary, OLD.error, OLD.done_published_at, OLD.request, OLD.effective_request)
         IS DISTINCT FROM ROW(NEW.state, NEW.started_at, NEW.finished_at, NEW.exit_code, NEW.cancel_requested, NEW.artifacts, NEW.run_summary, NEW.error, NEW.done_published_at, NEW.request, NEW.effective_request)
        THEN RAISE EXCEPTION 'dispatch projection requires operational authority'; END IF; RETURN NEW; END $$
        """)
    op.execute(
        "CREATE TRIGGER protect_job_authority BEFORE UPDATE ON jobs FOR EACH ROW EXECUTE FUNCTION tp_protect_job_authority()"
    )
    op.execute(
        "CREATE TRIGGER protect_dispatch_plans BEFORE UPDATE OR DELETE ON dispatch_plans FOR EACH ROW EXECUTE FUNCTION tp_reject_evidence_mutation()"
    )
    op.execute(
        "CREATE TRIGGER protect_operational_records BEFORE UPDATE OR DELETE ON operational_records FOR EACH ROW EXECUTE FUNCTION tp_reject_evidence_mutation()"
    )
    op.execute(
        "CREATE TRIGGER protect_committed_generations BEFORE UPDATE OR DELETE ON committed_generations FOR EACH ROW EXECUTE FUNCTION tp_reject_evidence_mutation()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER protect_job_authority ON jobs")
    op.drop_table("committed_generations")
    op.drop_table("operational_outbox")
    op.drop_table("operational_records")
    op.drop_table("dispatch_attempts")
    op.drop_table("dispatch_plans")
    op.drop_table("admission_capacity")
    op.execute("DROP FUNCTION tp_protect_job_authority()")
    op.execute("DROP FUNCTION tp_protect_dispatch_identity()")
    op.execute("DROP FUNCTION tp_reject_evidence_mutation()")
    op.execute("DROP SEQUENCE dispatch_lease_epoch_seq")
