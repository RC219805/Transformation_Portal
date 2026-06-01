from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest
import yaml
from PIL import Image

from transformation_portal.presence_security import (
    PresenceParameters,
    add_dither,
    embed_lsb_rgb,
    manifest_session_from_lsb,
    randomized_eye_line,
    randomized_prompts,
    sha3_manifest_hex,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA_PATH = PROJECT_ROOT / "docs/schemas/presence/tp.presence.manifest.v1_2/manifest.schema.json"
MANIFEST_EXAMPLE_PATH = PROJECT_ROOT / "docs/contracts/examples/tp.presence.manifest.v1_2.example.json"
TRUST_SCHEMA_PATH = PROJECT_ROOT / "docs/schemas/presence/tp.presence.trust_registry.v1_2/trust_registry.schema.json"
TRUST_EXAMPLE_PATH = PROJECT_ROOT / "docs/contracts/examples/tp.presence.trust_registry.v1_2.example.json"
RANDOMIZATION_PATH = PROJECT_ROOT / "config/presence_security/v1_2/randomization.yml"
CERTIFICATION_PATH = PROJECT_ROOT / "config/presence_security/v1_2/certification.yml"
LOCALES_PATH = PROJECT_ROOT / "config/presence_security/v1_2/locales.yml"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pythonpath = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return subprocess.run(
        [sys.executable, "-m", "transformation_portal.presence_security", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_presence_parameters_are_deterministic_and_locale_bounded() -> None:
    first = PresenceParameters("demo-session", "US_EN")
    second = PresenceParameters("demo-session", "US_EN")

    assert first.eye_line() == second.eye_line()
    assert first.blend_weights() == second.blend_weights()
    assert 0.26 <= first.eye_line() <= 0.28
    assert first.blend_weights()[0] == first.blend_weights()[2]


def test_presence_parameters_unknown_locale_falls_back_to_us_en() -> None:
    fallback = PresenceParameters("demo-session", "UNKNOWN")
    expected = PresenceParameters("demo-session", "US_EN")

    assert fallback.locale == "US_EN"
    assert fallback.eye_line() == expected.eye_line()
    assert randomized_eye_line("demo-session", "UNKNOWN") == expected.eye_line()


def test_prompt_order_is_sessionized_and_stable() -> None:
    prompts = ["Silent yes", "What would you do?", "Stay with me"]

    assert randomized_prompts(prompts, "demo-session") == randomized_prompts(prompts, "demo-session")
    assert sorted(randomized_prompts(prompts, "demo-session")) == sorted(prompts)


def test_add_dither_preserves_image_shape() -> None:
    source = Image.fromarray(np.full((4, 5, 3), 128, dtype=np.uint8), mode="RGB")

    result = add_dither(source, sigma=0.003, seed=7)

    assert result.mode == "RGB"
    assert result.size == source.size
    assert np.array(result).shape == np.array(source).shape


def test_lsb_watermark_roundtrip_extracts_manifest_and_session_ids() -> None:
    source = Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8), mode="RGB")
    manifest_hash = sha3_manifest_hex(b'{"presence": true}')

    watermarked = embed_lsb_rgb(source, manifest_hash, "demo-session")
    manifest_hash16, session_id16 = manifest_session_from_lsb(watermarked)

    assert manifest_hash16 == bytes.fromhex(manifest_hash)[:16]
    assert session_id16 == hashlib.sha256(b"demo-session").digest()[:16]


@pytest.mark.parametrize(
    ("schema_path", "example_path"),
    [
        (MANIFEST_SCHEMA_PATH, MANIFEST_EXAMPLE_PATH),
        (TRUST_SCHEMA_PATH, TRUST_EXAMPLE_PATH),
    ],
)
def test_presence_schema_examples_validate(schema_path: Path, example_path: Path) -> None:
    schema = _load_json(schema_path)
    example = _load_json(example_path)

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(example)


def test_presence_yaml_config_shapes_are_structured() -> None:
    randomization = yaml.safe_load(RANDOMIZATION_PATH.read_text(encoding="utf-8"))
    certification = yaml.safe_load(CERTIFICATION_PATH.read_text(encoding="utf-8"))
    locales = yaml.safe_load(LOCALES_PATH.read_text(encoding="utf-8"))

    assert randomization["randomization"]["micro_median_weights"][0] == {"min": 0.65, "max": 0.75}
    assert randomization["randomization"]["dither_sigma"] == {"min": 0.002, "max": 0.0045}
    assert certification["carolwood_certified"]["verification_levels"]["gold"]
    assert "US_EN" in locales["profiles"]


def test_presence_cli_params_outputs_json() -> None:
    result = _run_module("params", "--session", "demo-session", "--locale", "US_EN")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert 0.26 <= payload["eye_line"] <= 0.28
    assert payload["blend_weights"][0] == payload["blend_weights"][2]


def test_presence_cli_anchor_writes_hash_payload(tmp_path: Path) -> None:
    hero = tmp_path / "hero.jpg"
    web = tmp_path / "web.jpg"
    out = tmp_path / "anchor.json"
    hero.write_bytes(b"hero")
    web.write_bytes(b"web")

    result = _run_module(
        "anchor",
        "--manifest",
        str(MANIFEST_EXAMPLE_PATH),
        "--hero",
        str(hero),
        "--web",
        str(web),
        "--out",
        str(out),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["manifest_sha3"] == hashlib.sha3_256(MANIFEST_EXAMPLE_PATH.read_bytes()).hexdigest()
    assert payload["hero_sha3"] == hashlib.sha3_256(b"hero").hexdigest()
    assert payload["web_sha3"] == hashlib.sha3_256(b"web").hexdigest()


def test_presence_cli_watermark_lsb_writes_extractable_image(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    out = tmp_path / "watermarked.png"
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8), mode="RGB").save(source)

    result = _run_module(
        "watermark",
        "--image",
        str(source),
        "--manifest",
        str(MANIFEST_EXAMPLE_PATH),
        "--session",
        "demo-session",
        "--mode",
        "lsb",
        "--out",
        str(out),
    )

    assert result.returncode == 0, result.stderr
    manifest_hash16, session_id16 = manifest_session_from_lsb(Image.open(out))
    assert manifest_hash16 == hashlib.sha3_256(MANIFEST_EXAMPLE_PATH.read_bytes()).digest()[:16]
    assert session_id16 == hashlib.sha256(b"demo-session").digest()[:16]
