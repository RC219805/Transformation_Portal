"""add operational audit events

Revision ID: 0002_operational_audit_events
Revises: 0001_initial
Create Date: 2026-06-04
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_operational_audit_events"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "operational_audit_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=True),
        sa.Column("job_id", sa.String(length=64), nullable=True),
        sa.Column(
            "actor",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "request_context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_operational_audit_events_action", "operational_audit_events", ["action"], unique=False)
    op.create_index("ix_operational_audit_events_created_at", "operational_audit_events", ["created_at"], unique=False)
    op.create_index("ix_operational_audit_events_decision", "operational_audit_events", ["decision"], unique=False)
    op.create_index("ix_operational_audit_events_job_id", "operational_audit_events", ["job_id"], unique=False)
    op.create_index("ix_operational_audit_events_tenant_id", "operational_audit_events", ["tenant_id"], unique=False)
    op.create_index(
        "ix_operational_audit_events_tenant_created",
        "operational_audit_events",
        ["tenant_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_operational_audit_events_tenant_created", table_name="operational_audit_events")
    op.drop_index("ix_operational_audit_events_tenant_id", table_name="operational_audit_events")
    op.drop_index("ix_operational_audit_events_job_id", table_name="operational_audit_events")
    op.drop_index("ix_operational_audit_events_decision", table_name="operational_audit_events")
    op.drop_index("ix_operational_audit_events_created_at", table_name="operational_audit_events")
    op.drop_index("ix_operational_audit_events_action", table_name="operational_audit_events")
    op.drop_table("operational_audit_events")
