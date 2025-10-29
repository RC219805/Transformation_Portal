#!/usr/bin/env python3
"""
presence_cli_v1_3.py - governance + measurement + consent gating (v1.3)

New in v1.3:
  • measure        - auto-estimate eye-line and side gutters from an image
  • verify-manifest- enforce consent gating (and optionally signature) before detection

Requires: Pillow, numpy, (optional) PyNaCl for signature verification
"""
import argparse
import json
import sys
import base64
import hashlib
from typing import Tuple, Dict
from PIL import Image
import numpy as np

# Optional Ed25519 verify
try:
    from nacl.signing import VerifyKey
    from nacl.exceptions import BadSignatureError
    NACL_OK = True
except ImportError:
    NACL_OK = False


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.digest()


def measure_image(path: str, aspect: str = "4:5") -> Dict[str, float]:
    """Heuristic measurement for eye-line and side gutters (no face model required).
       - Eye-line: finds strongest horizontal gradient row in the central band in a plausible range,
         then blends with the target prior (0.27 for 4:5; 0.36 for 2:3) for robustness.
       - Gutters: inspects a lower row (0.72H for 4:5; 0.78H for 2:3), detects left/right edges.
    """
    im = Image.open(path).convert("L")
    w, h = im.size
    arr = np.array(im, dtype=np.float32) / 255.0

    # Eye-line search window and prior
    if aspect == "4:5":
        y0, y1, prior = int(0.22 * h), int(0.52 * h), 0.27
        chest_row = int(0.72 * h)
    else:  # 2:3
        y0, y1, prior = int(0.25 * h), int(0.55 * h), 0.36
        chest_row = int(0.78 * h)

    # Vertical gradient magnitude (row energy) in central 40% width
    cw = int(0.40 * w)
    x0 = (w - cw) // 2
    x1 = x0 + cw
    dy = np.abs(np.diff(arr, axis=0))
    row_energy = dy[y0:y1, x0:x1].sum(axis=1)
    idx = int(np.argmax(row_energy)) + y0
    eye_pct_raw = idx / h
    # blend with prior to stabilize: weighted toward measured when confident
    # confidence proxy = prominence over median
    # Small epsilon to avoid division by zero in normalization
    EPSILON = 1e-6
    # Minimum and maximum blend weights for measured value
    ALPHA_MIN = 0.3      # Lower bound for confidence weight
    ALPHA_MAX = 0.85     # Upper bound for confidence weight
    ALPHA_BASE = 0.45    # Base confidence weight
    ALPHA_PROM_SCALE = 0.08  # Scale factor for prominence influence
    prom = (row_energy.max() - np.median(row_energy)) / (np.std(row_energy) + EPSILON)
    alpha = max(ALPHA_MIN, min(ALPHA_MAX, ALPHA_BASE + ALPHA_PROM_SCALE * prom))  # Blend weight for measured vs prior
    eye_pct = alpha * eye_pct_raw + (1 - alpha) * prior

    # Gutters via horizontal gradient at chest row
    row = arr[chest_row, :]
    dx = np.abs(np.diff(row))
    thr = dx.mean() + 1.2 * dx.std()
    # left edge from left→center
    left_edge = None
    for x in range(5, w // 2):
        if dx[x] > thr:
            left_edge = x
            break
    # right edge from right→center
    right_edge = None
    for x in range(w - 6, w // 2, -1):
        if dx[x] > thr:
            right_edge = x
            break
    # Fallbacks if not found
    if left_edge is None:
        left_edge = int(0.14 * w)
    if right_edge is None:
        right_edge = int(0.86 * w)

    left_gutter_pct = left_edge / w
    right_gutter_pct = (w - right_edge) / w

    # Confidence heuristic
    conf_eye = max(0.0, min(1.0, prom / 6.0))
    conf_gut = 1.0 if (0.05 < left_gutter_pct < 0.35 and 0.05 < right_gutter_pct < 0.35) else 0.6

    return {
        "width": w, "height": h, "aspect": aspect,
        "eye_line_pct": round(float(eye_pct), 4),
        "eye_line_pct_raw": round(float(eye_pct_raw), 4),
        "eye_confidence": round(float(conf_eye), 3),
        "left_gutter_pct": round(float(left_gutter_pct), 4),
        "right_gutter_pct": round(float(right_gutter_pct), 4),
        "gutters_confidence": round(float(conf_gut), 3),
        "chest_row_px": chest_row
    }


def verify_manifest(manifest_path: str, hero: str = None, web: str = None,
                    public_key: str = None, signature_path: str = None,
                    require_signature: bool = False) -> Tuple[bool, str]:
    """Consent gating + optional signature verification.
       PASS only if: consent.status == 'granted' AND 'detect' in consent.scope.
       If require_signature, also verifies Ed25519 signature over JSON payload of sha256 files.
    """
    with open(manifest_path, "r") as f:
        m = json.load(f)
    consent = m.get("consent", {})
    if consent.get("status") != "granted":
        return False, "Consent not granted"
    scope = consent.get("scope") or []
    if "detect" not in scope:
        return False, "Scope does not allow detection"

    if require_signature:
        if not (public_key and signature_path and hero and web):
            return False, "Signature required but parameters missing"
        if not NACL_OK:
            return False, "PyNaCl not available for signature verification"
        # Build payload (sha256 of files; sorted JSON)
        payload = {
            "manifest_sha256": base64.b64encode(_sha256_file(manifest_path)).decode(),
            "hero_sha256": base64.b64encode(_sha256_file(hero)).decode(),
            "web_sha256": base64.b64encode(_sha256_file(web)).decode(),
        }
        ser = json.dumps(payload, sort_keys=True).encode("utf-8")
        with open(signature_path, "rb") as sig_file:
            sig = base64.b64decode(sig_file.read())
        with open(public_key, "rb") as pub_file:
            vk = VerifyKey(pub_file.read())
        try:
            vk.verify(ser, sig)
        except BadSignatureError:
            return False, "Signature invalid"
    return True, "PASS"


def main():
    ap = argparse.ArgumentParser(prog="presence-cli-v1.3")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="Auto-measure eye-line and side gutters from an image")
    m.add_argument("--image", required=True)
    m.add_argument("--aspect", choices=["4:5", "2:3"], default="4:5")
    m.add_argument("--out", help="Write JSON report")

    v = sub.add_parser("verify-manifest", help="Consent gating (and optional signature verify)")
    v.add_argument("--manifest", required=True)
    v.add_argument("--hero")
    v.add_argument("--web")
    v.add_argument("--public")           # Ed25519 public key
    v.add_argument("--signature")        # signature.b64
    v.add_argument("--require-signature", action="store_true")

    args = ap.parse_args()

    if args.cmd == "measure":
        report = measure_image(args.image, args.aspect)
        txt = json.dumps(report, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(txt)
            print("Wrote", args.out)
        else:
            print(txt)

    elif args.cmd == "verify-manifest":
        ok, msg = verify_manifest(args.manifest, args.hero, args.web, args.public, args.signature, args.require_signature)
        if ok:
            print("PASS:", msg)
            sys.exit(0)
        else:
            print("FAIL:", msg)
            sys.exit(2)


if __name__ == "__main__":
    main()
