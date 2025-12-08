from lux_depth_v2.validation.benchmark.dataset_registry import get_dataset_spec


def test_validation_v1_dirs_exist():
    spec = get_dataset_spec("validation_v1")

    assert spec.root_dir.exists()
    assert spec.input_dir.exists()
    assert spec.metadata_file.exists()

    assert spec.baselines_dir.exists()
    for _, p in spec.baselines.items():
        assert p.exists()
