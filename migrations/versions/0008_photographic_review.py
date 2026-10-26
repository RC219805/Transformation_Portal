"""Durable hash-bound photographic review state and mutation history.

Revision ID: 0008_photographic_review
Revises: 0007_photography_bindings
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# Alembic exposes these operation methods through its dynamic migration proxy.
# pylint: disable=no-member

revision = "0008_photographic_review"
down_revision = "0007_photography_bindings"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "photographic_review_states",
        sa.Column("job_id", sa.String(64), sa.ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
    )
    op.create_table(
        "photographic_review_events",
        sa.Column("job_id", sa.String(64), sa.ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("version", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("canonical_bytes", sa.LargeBinary(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
    )


def downgrade():
    op.drop_table("photographic_review_events")
    op.drop_table("photographic_review_states")
