"""Loopback-only real photographic API fixture for frontdoor browser tests.

Run with the repo interpreter and --fixture-only --port 10007. It uses a
public test API key, memory jobs and private disposable storage; never point it
at production services. Original pixels are explicit synthetic fixture inputs,
not a claim of immutable managed-admission coverage (covered by adapter tests).
All review/revision/export/packaging handlers and the frontdoor proxy are real.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path


def run(port: int) -> None:
    if not 1024 <= port <= 65535:
        raise ValueError("Fixture port must be unprivileged")
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    with tempfile.TemporaryDirectory(prefix="tp-photographic-browser-") as directory:
        root = Path(directory).resolve()
        os.environ.update(
            {
                "TP_API_KEY": "frontdoor-browser-smoke",
                "TP_ORCHESTRATOR_STATE_BACKEND": "memory",
                "TP_ORCHESTRATOR_QUEUE_BACKEND": "memory",
                "TP_ORCHESTRATOR_IN_PROCESS_WORKERS_ENABLED": "0",
                "TP_ARTIFACT_STORE": "local",
                "TP_ARTIFACT_LOCAL_ROOT": str(root / "artifacts"),
                "TP_PILOT_CONTROL_PLANE_ENABLED": "0",
                "TP_PORTAL_REVIEW_SURFACE_DEFERRED_ENABLED": "1",
            }
        )
        import numpy as np
        import tifffile
        import uvicorn
        from PIL import Image

        import app
        from transformation_portal.lux_depth_v4.photography import output_srgb_icc
        from transformation_portal.portal.photographic_review_access import AppReviewAccess

        height, width = 1536, 2048
        pixels = np.empty((height, width, 3), dtype=np.uint16)
        pixels[..., 0] = np.arange(width, dtype=np.uint16)[None, :] * 28
        pixels[..., 1] = np.arange(height, dtype=np.uint16)[:, None] * 36
        pixels[..., 2] = 14000
        pixels[300:800, 1200:1500] = [55000, 57000, 60000]
        final = root / "output/v6/input-0000/delivery.tif"
        final.parent.mkdir(parents=True)
        tifffile.imwrite(
            final,
            pixels,
            photometric="rgb",
            metadata=None,
            extratags=[(34675, "B", len(output_srgb_icc()), output_srgb_icc(), False)],
        )
        original = root / "original.png"
        with Image.fromarray((pixels / 257).astype(np.uint8)) as image:
            image.save(original, icc_profile=output_srgb_icc())
            image.thumbnail((1600, 1600))
            image.save(final.with_name("preview.png"), icc_profile=output_srgb_icc())
        now = time.time()
        job = app.Job(
            id="job_photographic_actual",
            created_at=now,
            started_at=now,
            finished_at=now,
            done_published_at=now,
            state="succeeded",
            progress=100,
            request={"pipeline": "lux-depth", "args": {"input_dir": str(root), "output_dir": str(root / "output")}},
            effective_request={"pipeline": "lux-depth", "args": {"input_dir": str(root), "output_dir": str(root / "output")}},
        )
        app._index_job_artifacts(job)
        # This fixture retains exact source identities even when the ordinary
        # legacy index deliberately skips hashing its >16 MiB native TIFF.
        final_relative = "v6/input-0000/delivery.tif"
        for item in job.artifacts.get("items", []):
            if (item.get("path") or item.get("relative_path")) == final_relative:
                with final.open("rb") as stream:
                    item["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                item["size_bytes"] = final.stat().st_size
                item["fingerprint_status"] = "ok"
        asyncio.run(app._job_repository().create(app._record_from_job(job)))
        app.JOBS[job.id] = job
        refs = []
        for path, public_path, role in (
            (final, "v6/input-0000/delivery.tif", "final"),
            (original, "original/input-0000.png", "original"),
        ):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            refs.append(
                {
                    "artifact_path": public_path,
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "scene_id": "input-0000",
                    "label": "Living room",
                    "source_name": path.name,
                    "role": role,
                    "kind": "indexed" if role == "final" else "fixture_original",
                    "input_color": "auto",
                }
            )

        async def fixture_originals(self, authorized_job, tenant):
            if authorized_job.id != job.id or tenant is not None:
                return [], "not_a_synthetic_fixture"
            return [
                {
                    **refs[1],
                    "kind": "original",
                    "source_name": "Living-room.png",
                    "label": "Living room",
                    "private_original_root": str(root),
                    "private_original_path": "original.png",
                }
            ], None

        # Only admission's original binding is synthetic. Actual app routes,
        # authentication, IO, review state and processors remain in use.
        AppReviewAccess._originals = fixture_originals
        print(f"Photographic API fixture ready: job_photographic_actual, loopback port {port}", flush=True)
        uvicorn.run(app.app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-only", action="store_true", required=True)
    parser.add_argument("--port", type=int, default=10007)
    run(parser.parse_args().port)
