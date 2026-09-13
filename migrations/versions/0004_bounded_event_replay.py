"""Add monotonic counters independent of retained event history.

Revision ID: 0004_bounded_event_replay
Revises: 0003_dispatch_authority
Create Date: 2026-09-13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# Alembic exposes these operation methods through its dynamic migration proxy.
# pylint: disable=no-member


revision: str = "0004_bounded_event_replay"
down_revision: str = "0003_dispatch_authority"
branch_labels: None = None
depends_on: None = None


def upgrade() -> None:
    op.create_table(
        "job_event_sequences",
        sa.Column("job_id", sa.String(length=64), primary_key=True),
        sa.Column("last_seq", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("last_seq >= 0", name="ck_job_event_sequence_nonnegative"),
    )
    # Stop writers during migration. Preserve every preexisting replay cursor;
    # the next append applies the configured retention cap to that job.
    op.execute("INSERT INTO job_event_sequences (job_id, last_seq) SELECT job_id, MAX(seq) FROM job_events GROUP BY job_id")
    op.alter_column("job_events", "seq", existing_type=sa.Integer(), type_=sa.BigInteger(), existing_nullable=False)


def downgrade() -> None:
    # Keep seq bigint: narrowing could truncate valid long-lived cursor values.
    op.drop_table("job_event_sequences")
