#!/usr/bin/env python3
"""Package explicitly selected, checksum-bound files without reprocessing them.

POSIX, Python 3.11+, standard library only. Supply an explicit manifest:

  {"schema":"tp.delivery.package_spec.v1","archive_root":"delivery-example",
   "files":[{"source":"example.txt","archive_path":"documents/example.txt",
   "sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
   "size_bytes":3}]}

This selects exactly the bytes ``abc`` (no newline). Source names are relative
to --source-root; archive paths are relative to archive_root. Save the spec as
.runtime/package-demo/spec.json, then run from the repository root:

  mkdir -p .runtime/package-demo/source .runtime/package-demo/output .runtime/package-demo/copy
  printf abc > .runtime/package-demo/source/example.txt
  python3 scripts/utilities/package_verified_delivery.py \\
    --manifest .runtime/package-demo/spec.json \\
    --source-root .runtime/package-demo/source \\
    --output .runtime/package-demo/output/example.zip \\
    --copy-to .runtime/package-demo/copy

The output parent and optional copy directory must already exist outside
source-root; copy-to must differ from the output parent.
Outputs are NEW.zip, NEW.zip.sha256, NEW.zip.receipt.json;
optional copy-to receives the same three verified files. Existing names are
never replaced. Private staging and full verification precede publication.
Each file is published atomically; the multi-file/cross-directory publication
is not crash-atomic. Recoverable publication failures roll back owned links.

The ZIP includes original selected bytes, _package/manifest.json (normalized
selection), and _package/SHA256SUMS.txt under archive_root. Checksum lines use
full ZIP member names and exclude the checksum file itself. From the directory
containing example.zip, verify and extract with:

  shasum -a 256 -c example.zip.sha256
  unzip example.zip
  shasum -a 256 -c delivery-example/_package/SHA256SUMS.txt

The manifest limit is 8 MiB and 1..10,000 entries. Paths must be portable,
normalized relative POSIX paths. Symlink roots/entries/destinations, nonregular
inputs, unsafe or colliding archive names, changed files and SHA/size mismatches
fail closed. Keep generated packages ignored, for example under .runtime/.

This checks byte integrity against caller-supplied hashes only. It does not
discover files, run models, reprocess images, or replay native generations.
Curated exports need their own mappings and are not automatically replayable.
It establishes no photographic, semantic, licensing, physical-depth, or
production acceptance. ZIP byte reproducibility is scoped to the same Python
and compression runtime. Success prints a JSON receipt; errors exit 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Preserve direct source execution with Python 3.11+ and no installed package or
# third-party dependency. Installed callers use the supported library directly.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformation_portal.core.verified_delivery import PackageError, build_package, main  # noqa: E402

__all__ = ["PackageError", "build_package", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
