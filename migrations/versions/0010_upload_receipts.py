"""Request-bound staged upload ownership and retained job references.

Revision ID: 0010_upload_receipts
Revises: 0009_job_submissions
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# pylint: disable=no-member
revision = "0010_upload_receipts"
down_revision = "0009_job_submissions"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "upload_receipts",
        sa.Column("batch_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("owner_sha256", sa.String(64), nullable=False),
        sa.Column("request_id", sa.String(36), nullable=False),
        sa.Column("generation", sa.String(32), nullable=False),
        sa.Column("batch_root", sa.Text(), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
        sa.Column("receipt_sha256", sa.String(64), nullable=True),
        sa.Column("cleanup_checked_at", sa.Float(), nullable=True),
        sa.Column("response_data", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint("state IN ('staging', 'ready', 'deleting', 'expired')", name="ck_upload_receipt_state"),
    )
    op.create_index(
        "uq_upload_receipt_owner_request", "upload_receipts", ["tenant_id", "owner_sha256", "request_id"], unique=True
    )
    op.create_index("ix_upload_receipts_expires_at", "upload_receipts", ["expires_at"])
    op.create_table(
        "upload_job_references",
        sa.Column("batch_id", sa.String(64), sa.ForeignKey("upload_receipts.batch_id"), primary_key=True),
        sa.Column("job_id", sa.String(64), sa.ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    )


def downgrade():
    # Application rollback keeps these additive records and readers. A schema
    # downgrade is an explicit operator action after outstanding work is drained.
    op.drop_table("upload_job_references")
    op.drop_index("ix_upload_receipts_expires_at", table_name="upload_receipts")
    op.drop_index("uq_upload_receipt_owner_request", table_name="upload_receipts")
    op.drop_table("upload_receipts")
