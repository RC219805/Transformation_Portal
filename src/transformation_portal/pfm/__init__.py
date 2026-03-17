"""Pipeline Foundation Model (PFM) package.

A foundation model for pipeline optimization that learns from
execution logs and generalizes across projects.

Components:
- data_pipeline: Log canonicalization and dataset creation
- tokenizer: Structured data to tensor conversion
- model: Graph + Temporal Transformer architecture
- multimodal: Image + 3D embedding integration
- trainer: Pretraining and fine-tuning utilities
"""

from transformation_portal.pfm.data_pipeline import (
    RunRecord,
    SequenceDataset,
    StepRecord,
    build_sequence,
)
from transformation_portal.pfm.tokenizer import PFMTokenizer

__all__ = [
    "RunRecord",
    "StepRecord",
    "SequenceDataset",
    "build_sequence",
    "PFMTokenizer",
]
