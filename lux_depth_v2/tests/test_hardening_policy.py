from pathlib import Path

import pytest

from lux_depth_v2.hardening.policy import HardeningPolicy


def test_output_root_enforcement(tmp_path: Path):
    root = tmp_path / "allowed"
    root.mkdir()
    policy = HardeningPolicy(enforce_output_within=str(root))

    ok = root / "out"
    ok.mkdir()
    policy.assert_output_allowed(ok)

    bad = tmp_path / "not_allowed"
    bad.mkdir()
    with pytest.raises(Exception):
        policy.assert_output_allowed(bad)
