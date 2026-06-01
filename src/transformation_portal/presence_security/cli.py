"""Command-line interface for Presence Security helpers."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image

from transformation_portal.ingest.canonical_json import dumps_json
from transformation_portal.presence_security.parameters import PresenceParameters
from transformation_portal.presence_security.watermarking import embed_dct_luma, embed_lsb_rgb


def sha256_file(path: Path) -> bytes:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.digest()


def anchor_payload(manifest_path: Path, hero_path: Path, web_path: Path) -> dict[str, str]:
    manifest_bytes = manifest_path.read_bytes()
    hero_bytes = hero_path.read_bytes()
    web_bytes = web_path.read_bytes()
    return {
        "assets_sha3": hashlib.sha3_256(hero_bytes + web_bytes).hexdigest(),
        "manifest_sha3": hashlib.sha3_256(manifest_bytes).hexdigest(),
        "hero_sha3": hashlib.sha3_256(hero_bytes).hexdigest(),
        "web_sha3": hashlib.sha3_256(web_bytes).hexdigest(),
    }


def cmd_anchor(args: argparse.Namespace) -> None:
    payload = anchor_payload(Path(args.manifest), Path(args.hero), Path(args.web))
    out_path = Path(args.out)
    out_path.write_text(dumps_json(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Anchor payload written:", out_path)


def cmd_watermark(args: argparse.Namespace) -> None:
    with Image.open(args.image) as handle:
        img = handle.convert("RGB")
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest_hash = hashlib.sha3_256(manifest_bytes).hexdigest()
    if args.mode == "lsb":
        out_img = embed_lsb_rgb(img, manifest_hash, args.session)
    else:
        out_img = embed_dct_luma(img, manifest_hash, args.session, strength=2.0)
    out_img.save(args.out)
    print("Watermarked ->", args.out)


def cmd_params(args: argparse.Namespace) -> None:
    pp = PresenceParameters(session_key=args.session, locale=args.locale)
    prompts = [prompt.strip() for prompt in args.prompts.split(",") if prompt.strip()]
    data = {
        "eye_line": pp.eye_line(),
        "blend_weights": pp.blend_weights(),
        "prompt_order": pp.prompt_order(prompts),
        "dither_sigma": pp.dither_sigma(),
    }
    print(dumps_json(data, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="presence-security")
    sub = parser.add_subparsers(dest="cmd", required=True)

    anchor = sub.add_parser("anchor", help="Write blockchain anchor payload hashes.")
    anchor.add_argument("--manifest", required=True)
    anchor.add_argument("--hero", required=True)
    anchor.add_argument("--web", required=True)
    anchor.add_argument("--out", default="anchor_payload.json")
    anchor.set_defaults(func=cmd_anchor)

    watermark = sub.add_parser("watermark", help="Embed watermark (lsb|dct) from manifest/session.")
    watermark.add_argument("--image", required=True)
    watermark.add_argument("--manifest", required=True)
    watermark.add_argument("--session", required=True)
    watermark.add_argument("--mode", choices=["lsb", "dct"], default="dct")
    watermark.add_argument("--out", required=True)
    watermark.set_defaults(func=cmd_watermark)

    params = sub.add_parser("params", help="Emit sessionized parameters.")
    params.add_argument("--session", required=True)
    params.add_argument("--locale", default="US_EN")
    params.add_argument("--prompts", default="Silent yes,What would you do?,Stay with me")
    params.set_defaults(func=cmd_params)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
