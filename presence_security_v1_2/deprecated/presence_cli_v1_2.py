#!/usr/bin/env python3
"""
presence_cli_v1_2.py — extended CLI (security v1.2)
Requires: Pillow, PyNaCl (for signing)
"""

import argparse
import hashlib
import json

from PIL import Image

from presence_security_v1_2.watermarking import embed_dct_luma, embed_lsb_rgb
from presence_security_v1_2.presence_params import PresenceParameters


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.digest()


def anchor_payload(manifest_path, hero_path, web_path):
    with open(manifest_path, "rb") as f:
        mbytes = f.read()
    h_m = hashlib.sha3_256(mbytes).hexdigest()
    with open(hero_path, "rb") as f:
        h_hero = hashlib.sha3_256(f.read()).hexdigest()
    with open(web_path, "rb") as f:
        h_web = hashlib.sha3_256(f.read()).hexdigest()
    payload = {"manifest_sha3": h_m, "hero_sha3": h_hero, "web_sha3": h_web}
    return payload


def cmd_anchor(args):
    payload = anchor_payload(args.manifest, args.hero, args.web)
    out_path = args.out or "anchor_payload.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print("Anchor payload written:", out_path)


def cmd_watermark(args):
    img = Image.open(args.image).convert("RGB")
    with open(args.manifest, "rb") as f:
        manifest_bytes = f.read()
    manifest_hash = hashlib.sha3_256(manifest_bytes).hexdigest()
    if args.mode == "lsb":
        out_img = embed_lsb_rgb(img, manifest_hash, args.session)
    else:
        out_img = embed_dct_luma(img, manifest_hash, args.session, strength=2.0)
    out_img.save(args.out)
    print("Watermarked ->", args.out)


def cmd_params(args):
    pp = PresenceParameters(session_key=args.session, locale=args.locale)
    data = {
        "eye_line": pp.eye_line(),
        "blend_weights": pp.blend_weights(),
        "prompt_order": pp.prompt_order(args.prompts.split(",")),
        "dither_sigma": pp.dither_sigma(),
    }
    print(json.dumps(data, indent=2))


def main():
    ap = argparse.ArgumentParser(prog="presence-cli-v1.2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("anchor", help="Write blockchain anchor payload (sha3)")
    a.add_argument("--manifest", required=True)
    a.add_argument("--hero", required=True)
    a.add_argument("--web", required=True)
    a.add_argument("--out", default="anchor_payload.json")
    a.set_defaults(func=cmd_anchor)

    w = sub.add_parser("watermark", help="Embed watermark (lsb|dct) from manifest/session")
    w.add_argument("--image", required=True)
    w.add_argument("--manifest", required=True)
    w.add_argument("--session", required=True)
    w.add_argument("--mode", choices=["lsb", "dct"], default="dct")
    w.add_argument("--out", required=True)
    w.set_defaults(func=cmd_watermark)

    p = sub.add_parser("params", help="Emit sessionized parameters")
    p.add_argument("--session", required=True)
    p.add_argument("--locale", default="US_EN")
    p.add_argument("--prompts", default="Silent yes,What would you do?,Stay with me")
    p.set_defaults(func=cmd_params)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
