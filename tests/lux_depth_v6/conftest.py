"""Shared upstream evidence fixtures for the opt-in V6 contracts."""

from tests.lux_depth_v5 import test_evidence as v5_evidence
from tests.lux_depth_v5 import test_pipeline as v5_pipeline

completed = v5_evidence.completed
request_case = v5_pipeline.request_case
