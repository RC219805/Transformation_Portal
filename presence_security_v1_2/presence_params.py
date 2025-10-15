# presence_params.py — sessionized parameter obfuscation & derivations (v1.2)
# Usage:
#   from presence_params import PresenceParameters
#   pp = PresenceParameters(session_key="any-unique-string", locale="US_EN")
#   eye = pp.eye_line()  # e.g., 0.2683 (within 0.26–0.28 for US_EN)
#   weights = pp.blend_weights()  # [0.71, 0.98, 0.71] randomized per session

import hashlib
import random

_LOCALE_BOUNDS = {
    "US_EN": (0.26, 0.28),
    "JP_JA": (0.27, 0.29),  # softer variation
    "DE_DE": (0.26, 0.28),
    "CN_ZH": (0.265, 0.285),
    "IN_EN": (0.26, 0.28),
    "GCC_AR": (0.265, 0.285),
}


class PresenceParameters:
    def __init__(self, session_key: str, locale: str = "US_EN"):
        sk = (session_key or "default").encode("utf-8")
        self.salt = hashlib.sha256(sk).digest()
        self.locale = locale if locale in _LOCALE_BOUNDS else "US_EN"
        # derive RNG
        seed = int.from_bytes(self.salt, "big")
        self._rng = random.Random(seed)

    def _interp(self, lo, hi, b2):
        return lo + (hi - lo) * (b2 / 65535.0)

    def eye_line(self) -> float:
        lo, hi = _LOCALE_BOUNDS[self.locale]
        # derive two bytes → 0..65535; map to [lo, hi]
        b2 = int.from_bytes(self.salt[0:2], "big")
        return self._interp(lo, hi, b2)

    def blend_weights(self):
        # micro-median weights for [t*-1, t*, t*+1], randomized but bounded
        seed = int.from_bytes(self.salt[2:4], "big")
        self._rng.seed(seed)
        w1 = 0.65 + (self._rng.randrange(0, 101) / 1000.0)  # 0.65–0.75
        w2 = 0.95 + (self._rng.randrange(0, 51) / 1000.0)   # 0.95–1.00
        return [round(w1, 3), round(w2, 3), round(w1, 3)]

    def prompt_order(self, prompts):
        # stable but sessionized shuffle
        p = list(prompts)
        self._rng.shuffle(p)
        return p

    def dither_sigma(self):
        # small controlled noise to resist ML fingerprinting patterns
        s = 0.002 + (int.from_bytes(self.salt[6:8], "big") % 6) * 0.0005  # 0.002–0.0045
        return round(s, 5)
