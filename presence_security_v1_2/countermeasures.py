# countermeasures.py — detection countermeasures & controlled randomness (v1.2)

import numpy as np
from PIL import Image

from presence_security_v1_2.presence_params import PresenceParameters


def add_dither(img: Image.Image, sigma: float, seed: int = 42) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))


def randomized_eye_line(session_key: str, locale: str = "US_EN") -> float:
    return PresenceParameters(session_key, locale).eye_line()


def randomized_blend_weights(session_key: str) -> list:
    return PresenceParameters(session_key).blend_weights()


def randomized_prompts(prompts: list, session_key: str) -> list:
    return PresenceParameters(session_key).prompt_order(prompts)
