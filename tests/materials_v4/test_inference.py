"""Experimental inference is bounded, source-bound, explicit and nonauthorizing."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.materials_v4 import inference, proposal, semantics
from transformation_portal.materials_v4.contracts import MaterialsError
from transformation_portal.materials_v4.engine import ResponsePolicy, apply_response, plan_response
from transformation_portal.materials_v4.inference import InferenceConfig, infer_materials, prepare_inference
from transformation_portal.materials_v4.proposal import Proposal
from transformation_portal.materials_v4.semantics import MATERIAL_PROMPTS, SemanticObservation, classify_scores, masked_crop

pytestmark = pytest.mark.unit


@pytest.fixture
def prepared_fixture(tmp_path, monkeypatch):
    save_file = pytest.importorskip("safetensors.numpy").save_file

    sam = tmp_path / "sam.pt"
    sam.write_bytes(b"fixture-sam")
    clip = tmp_path / "clip.safetensors"
    save_file({"tensor": np.zeros((2,), np.float32)}, str(clip))
    monkeypatch.setattr(inference, "SAM2_LARGE_SHA256", hashlib.sha256(sam.read_bytes()).hexdigest())
    monkeypatch.setattr(inference, "_runtime_payload", lambda device: {"actual_device": device, "version": "test"})
    master = ImageMaster(np.full((28, 42, 3), 0.5, np.float32), "a" * 64, 16)
    config = InferenceConfig(sam, clip, proxy_longest_side=28)
    prepared = prepare_inference(master, config)
    return master, prepared


def test_prepared_inference_preserves_master_and_cannot_authorize_edits(prepared_fixture, monkeypatch):
    master, prepared = prepared_fixture
    monkeypatch.setattr(
        inference, "generate_proposals", lambda image, *args, **kwargs: (Proposal(np.ones(image.shape[:2], bool), 0.99, 0.98),)
    )
    monkeypatch.setattr(
        inference, "classify_proposals", lambda *args, **kwargs: (SemanticObservation("water", 0.999, 0.8, 0.9, None),)
    )
    evidence = infer_materials(master, prepared)
    assert evidence.source_sha256 == master.source_sha256
    assert evidence.shape == master.shape
    assert evidence.calibration is None
    assert evidence.regions[0].geometric_quality == 0.99
    assert evidence.regions[0].score_type == "uncalibrated_clip_ranking_v1"
    assert evidence.producer["cache_policy"] == "off"
    policy = ResponsePolicy(min_coverage_px=1, min_confidence=0.0)
    plan = plan_response(master, evidence, policy)
    result, receipt = apply_response(master, evidence, plan)
    np.testing.assert_array_equal(result.pixels, master.pixels)
    assert all(item.status != "apply" for item in plan.regions)


def test_mutated_weights_fail_before_any_model_runs(prepared_fixture, monkeypatch):
    master, prepared = prepared_fixture
    prepared.config.sam2_checkpoint.write_bytes(b"changed-model")
    monkeypatch.setattr(
        inference, "generate_proposals", lambda *args, **kwargs: pytest.fail("Model initialized despite changed weights")
    )
    with pytest.raises(MaterialsError, match="weight bytes changed"):
        infer_materials(master, prepared)


def test_changed_master_and_runtime_fail_before_model_runs(prepared_fixture, monkeypatch):
    master, prepared = prepared_fixture
    other = ImageMaster(master.pixels + 0.1, master.source_sha256, 16)
    with pytest.raises(MaterialsError, match="master differs"):
        infer_materials(other, prepared)
    monkeypatch.setattr(inference, "_runtime_payload", lambda device: {"actual_device": device, "version": "changed"})
    with pytest.raises(MaterialsError, match="runtime changed"):
        infer_materials(master, prepared)


def test_cancelled_execution_has_no_model_or_artifact_side_effects(prepared_fixture, monkeypatch):
    master, prepared = prepared_fixture
    monkeypatch.setattr(inference, "generate_proposals", lambda *args, **kwargs: pytest.fail("Cancelled inference started"))
    with pytest.raises(MaterialsError, match="cancelled"):
        infer_materials(master, prepared, cancelled=lambda: True)


def test_post_inference_model_mutation_cannot_emit_evidence(prepared_fixture, monkeypatch):
    master, prepared = prepared_fixture
    monkeypatch.setattr(inference, "generate_proposals", lambda image, *args, **kwargs: ())

    def mutate(*args, **kwargs):
        prepared.config.clip_checkpoint.write_bytes(b"changed-during-execution")
        return ()

    monkeypatch.setattr(inference, "classify_proposals", mutate)
    with pytest.raises(MaterialsError, match="weight bytes changed"):
        infer_materials(master, prepared)


def test_temporary_source_weight_swap_cannot_change_consumed_model_bytes(prepared_fixture, monkeypatch):
    master, prepared = prepared_fixture
    paths = []

    def proposals(image, config, **kwargs):
        assert config.sam2_checkpoint.suffix == ".pt"
        paths.append(config.sam2_checkpoint)
        original = prepared.config.sam2_checkpoint.read_bytes()
        prepared.config.sam2_checkpoint.write_bytes(b"temporary-substitution")
        try:
            assert config.sam2_checkpoint != prepared.config.sam2_checkpoint
            assert config.sam2_checkpoint.read_bytes() == original
        finally:
            prepared.config.sam2_checkpoint.write_bytes(original)
        return ()

    def semantics(image, proposals, config, **kwargs):
        assert config.clip_checkpoint.suffix == ".safetensors"
        paths.append(config.clip_checkpoint)
        original = prepared.config.clip_checkpoint.read_bytes()
        prepared.config.clip_checkpoint.write_bytes(b"temporary-clip-substitution")
        try:
            assert config.clip_checkpoint != prepared.config.clip_checkpoint
            assert config.clip_checkpoint.read_bytes() == original
        finally:
            prepared.config.clip_checkpoint.write_bytes(original)
        return ()

    monkeypatch.setattr(inference, "generate_proposals", proposals)
    monkeypatch.setattr(inference, "classify_proposals", semantics)
    observed = infer_materials(master, prepared)
    assert observed.status == "available" and observed.regions == ()
    assert all(not path.exists() for path in paths)


def test_mask_restoration_budget_is_checked_before_lifting(prepared_fixture, monkeypatch):
    master, prepared = prepared_fixture
    prepared = replace(prepared, config=replace(prepared.config, max_mask_bytes=1))
    monkeypatch.setattr(
        inference, "generate_proposals", lambda image, *args, **kwargs: (Proposal(np.ones(image.shape[:2], bool), 0.9, 0.9),)
    )
    monkeypatch.setattr(
        inference, "classify_proposals", lambda *args, **kwargs: (SemanticObservation("water", 0.9, 0.8, 0.8, None),)
    )
    with pytest.raises(MaterialsError, match="mask budget"):
        infer_materials(master, prepared)


def test_missing_local_weight_and_nonpinned_sam_fail_closed(tmp_path, monkeypatch):
    master = ImageMaster(np.ones((14, 14, 3), np.float32), "a" * 64, 16)
    with pytest.raises(MaterialsError):
        prepare_inference(master, InferenceConfig(tmp_path / "missing.pt", tmp_path / "missing.safetensors"))


def test_masked_semantics_distinguishes_regions_with_identical_boxes():
    image = np.full((4, 4, 3), 200, np.uint8)
    diagonal = np.eye(4, dtype=bool)
    first = np.asarray(masked_crop(image, diagonal))
    second = np.asarray(masked_crop(image, diagonal[:, ::-1]))
    assert first.shape == second.shape
    assert not np.array_equal(first, second)
    assert np.all(first[~diagonal] == 0)


def test_unknown_rejection_prevents_high_relative_negative_similarity_promotion():
    config = SimpleNamespace(min_similarity=0.2, min_top_probability=0.5, min_margin=0.05)
    scores = np.full(len(MATERIAL_PROMPTS), -0.4)
    scores[2] = -0.1
    observed = classify_scores(scores, config)
    assert observed.confidence > 0.9
    assert observed.label == "unknown"
    assert observed.rejection == "below_absolute_similarity"
    tied = classify_scores(np.full(len(MATERIAL_PROMPTS), 0.8), config)
    assert tied.label == "unknown"


def test_uncompressed_rle_rejects_hostile_lengths_before_allocation(monkeypatch):
    monkeypatch.setattr(proposal.np, "zeros", lambda *args, **kwargs: pytest.fail("Allocation before RLE validation"))
    with pytest.raises(MaterialsError):
        proposal._decode_rle({"size": [4, 4], "counts": [0, 10**12]}, (4, 4))


def test_rle_restores_column_major_sam_geometry():
    mask = proposal._decode_rle({"size": [2, 3], "counts": [1, 2, 3]}, (2, 3))
    np.testing.assert_array_equal(mask, [[False, True, False], [True, False, False]])


def test_sam_candidate_scratch_fails_before_loading_model(tmp_path, monkeypatch):
    config = InferenceConfig(tmp_path / "missing.pt", tmp_path / "missing.safetensors", max_candidate_bytes=1)
    monkeypatch.setattr(
        inference, "_file_identity", lambda *args, **kwargs: pytest.fail("Read model before scratch preflight")
    )
    master = ImageMaster(np.ones((14, 14, 3), np.float32), "a" * 64, 16)
    with pytest.raises(MaterialsError, match="scratch estimate.*before batch"):
        prepare_inference(master, config)
    with pytest.raises(MaterialsError, match="reduce proxy_longest_side"):
        proposal.generate_proposals(np.ones((14, 14, 3), np.uint8), config, master_shape=(14, 14))


def test_sam_candidate_accumulator_budget_prevents_next_batch():
    config = SimpleNamespace(max_candidate_bytes=1_000_000_000, max_proposals=8)
    budget = proposal._CandidateBudget((14, 14), config)
    budget.limit = 3 * budget.per_candidate_scratch
    budget.before_batch(1)
    budget.admit_batch([{"size": [14, 14], "counts": [196]}])
    with pytest.raises(MaterialsError, match="scratch estimate.*before batch"):
        budget.before_batch(1)


def test_sam_pre_nms_limit_is_checked_before_upstream_accumulation(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    sam_generator = pytest.importorskip("sam2.automatic_mask_generator")
    sam_builder = pytest.importorskip("sam2.build_sam")
    accumulated, batches = [], []

    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        def _process_batch(self, points):
            batches.append(len(points))
            return {"rles": [{"size": [14, 14], "counts": [196]}]}

        def generate(self, image):
            for _ in range(2):
                result = self._process_batch(np.zeros((1, 2)))
                accumulated.extend(result["rles"])
            pytest.fail("Oversized candidate accumulator completed")

    monkeypatch.setattr(sam_generator, "SAM2AutomaticMaskGenerator", FakeGenerator)
    monkeypatch.setattr(
        sam_builder,
        "build_sam2",
        lambda *args, **kwargs: SimpleNamespace(image_size=1024, parameters=lambda: iter([torch.zeros(1)])),
    )
    config = InferenceConfig(tmp_path / "sam.pt", tmp_path / "clip.safetensors", max_proposals=1)
    with pytest.raises(MaterialsError, match="pre-NMS candidate count.*before accumulation"):
        proposal.generate_proposals(np.ones((14, 14, 3), np.uint8), config, master_shape=(14, 14))
    assert batches == [1, 1]
    assert len(accumulated) == 1


def test_classifier_runs_bounded_microbatches_without_remote_weights(tmp_path, monkeypatch):
    open_clip = pytest.importorskip("open_clip")
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    batches = []

    class FakeModel:
        def load_state_dict(self, state, strict):
            assert strict is True

        def eval(self):
            return self

        def parameters(self):
            return iter([torch.zeros(1)])

        def encode_text(self, tokens):
            return tokens

        def encode_image(self, tensors):
            batches.append(len(tensors))
            values = torch.zeros((len(tensors), len(MATERIAL_PROMPTS)))
            values[:, 2] = 1
            return values

    def create(*args, **kwargs):
        assert kwargs["pretrained"] is None
        assert kwargs["pretrained_image"] is False and kwargs["pretrained_text"] is False
        return FakeModel(), None, lambda crop: torch.zeros((3, 4, 4))

    monkeypatch.setattr(open_clip, "create_model_and_transforms", create)
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda name: lambda prompts: torch.eye(len(prompts)))
    monkeypatch.setattr(safetensors_torch, "load_file", lambda *args, **kwargs: {})
    config = InferenceConfig(tmp_path / "sam.pt", tmp_path / "clip.safetensors", classifier_batch_size=2)
    proposals = tuple(Proposal(np.ones((4, 4), bool), 0.9, 0.9) for _ in range(5))
    observed = semantics.classify_proposals(np.ones((4, 4, 3), np.uint8), proposals, config)
    assert batches == [2, 2, 1]
    assert len(observed) == 5
    assert all(item.label == "water" for item in observed)


@pytest.mark.parametrize(
    "options",
    [
        {"device": "auto"},
        {"crop_n_layers": 1},
        {"classifier_batch_size": 0},
        {"proxy_longest_side": 10000},
        {"min_margin": True},
    ],
)
def test_unbounded_or_implicit_inference_options_are_rejected(tmp_path, options):
    with pytest.raises(MaterialsError):
        InferenceConfig(tmp_path / "sam.pt", tmp_path / "clip.safetensors", **options)
