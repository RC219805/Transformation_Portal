"""CLI regression tests for tools/archive_manifest_reports.py."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "archive_manifest_reports.py"
HAS_TOOL_DEPS = importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("pandas") is not None
RUNNING_IN_CI = os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}

pytestmark = [pytest.mark.regression]

MANIFEST_COLUMNS = [
    "SourceFile",
    "FileName",
    "FileSize",
    "Model",
    "DateTimeOriginal",
    "ImageSize",
    "Quality",
    "FocalLength",
    "ShutterSpeed",
    "Aperture",
    "ISO",
    "WhiteBalance",
    "Flash",
]


class ArchiveManifestReportsCliTest(unittest.TestCase):
    """Validate deterministic report generation and root-level path handling."""

    def setUp(self) -> None:
        if HAS_TOOL_DEPS:
            return
        msg = "archive_manifest_reports requires numpy and pandas (install requirements/tools-archive.txt)."
        if RUNNING_IN_CI:
            self.fail(f"{msg} CI must install tool dependencies so these regression tests are not silently skipped.")
        self.skipTest(msg)

    def _run_cli(
        self, input_csv: Path, outdir: Path, *, extra_args: list[str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            sys.executable,
            str(TOOL_PATH),
            "--input",
            str(input_csv),
            "--outdir",
            str(outdir),
            "--no-by-drive",
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    def _load_tool_module(self):
        spec = importlib.util.spec_from_file_location("archive_manifest_reports_tool", TOOL_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load archive_manifest_reports module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _write_manifest(self, manifest_csv: Path, rows: list[dict[str, str]]) -> None:
        with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv_gz_rows(self, path: Path) -> list[dict[str, str]]:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_root_level_paths_use_empty_dir_and_schema_stays_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "IMG_0001.CR2",
                        "FileName": "IMG_0001.CR2",
                        "FileSize": "10 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                    {
                        "SourceFile": "IMG_0001.XMP",
                        "FileName": "IMG_0001.XMP",
                        "FileSize": "1 KB",
                        "Model": "",
                        "DateTimeOriginal": "",
                        "ImageSize": "",
                        "Quality": "",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                    {
                        "SourceFile": "/vault/All Archive/DriveA/Part1/sub/IMG_0002.JPG",
                        "FileName": "IMG_0002.JPG",
                        "FileSize": "2 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:01:00",
                        "ImageSize": "100x100",
                        "Quality": "",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                ],
            )

            result = self._run_cli(manifest_csv, outdir)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            archive_rows = self._read_csv_gz_rows(outdir / "archive_index_normalized.csv.gz")
            root_row = next(row for row in archive_rows if row["relpath"] == "IMG_0001.CR2")
            self.assertEqual(root_row["dir_within_drive"], "")
            self.assertEqual(root_row["basename"], "IMG_0001")
            self.assertNotIn("is_primary_raw", root_row)
            self.assertNotIn("is_jpeg", root_row)
            self.assertIn("filesize_approx_bytes_decimal", root_row)

            asset_rows = self._read_csv_gz_rows(outdir / "asset_grouping_report.csv.gz")
            root_asset = next(row for row in asset_rows if row["basekey"] == "IMG_0001")
            self.assertEqual(root_asset["origin_drive"], "")
            self.assertEqual(root_asset["dir_rel"], "")
            self.assertEqual(root_asset["n_raw_files"], "1")
            self.assertEqual(root_asset["n_xmp"], "1")

    def test_all_root_level_paths_do_not_raise_dir_split_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "IMG_1001.CR2",
                        "FileName": "IMG_1001.CR2",
                        "FileSize": "10 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                    {
                        "SourceFile": "IMG_1002.JPG",
                        "FileName": "IMG_1002.JPG",
                        "FileSize": "2 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:01:00",
                        "ImageSize": "100x100",
                        "Quality": "",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                ],
            )

            result = self._run_cli(manifest_csv, outdir)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            archive_rows = self._read_csv_gz_rows(outdir / "archive_index_normalized.csv.gz")
            self.assertTrue(all(row["dir_within_drive"] == "" for row in archive_rows))

    def test_outputs_are_deterministic_and_emit_expected_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            out_a = temp / "out_a"
            out_b = temp / "out_b"

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "/vault/All Archive/DriveA/Part1/vid_set/CLIP001.JPG",
                        "FileName": "CLIP001.JPG",
                        "FileSize": "2 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "",
                        "ImageSize": "100x100",
                        "Quality": "",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                    {
                        "SourceFile": "/vault/All Archive/DriveA/Part1/xmp_only/ORPHAN.XMP",
                        "FileName": "ORPHAN.XMP",
                        "FileSize": "1 KB",
                        "Model": "",
                        "DateTimeOriginal": "",
                        "ImageSize": "",
                        "Quality": "",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                    {
                        "SourceFile": "/vault/All Archive/DriveA/Part1/vid_set/CLIP001.MOV",
                        "FileName": "CLIP001.MOV",
                        "FileSize": "100 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "",
                        "ImageSize": "",
                        "Quality": "",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                ],
            )

            first = self._run_cli(manifest_csv, out_a)
            second = self._run_cli(manifest_csv, out_b)
            self.assertEqual(first.returncode, 0, msg=first.stderr)
            self.assertEqual(second.returncode, 0, msg=second.stderr)

            archive_a = (out_a / "archive_index_normalized.csv.gz").read_bytes()
            archive_b = (out_b / "archive_index_normalized.csv.gz").read_bytes()
            self.assertEqual(archive_a, archive_b)
            self.assertEqual(int.from_bytes(archive_a[4:8], "little"), 0)
            self.assertEqual(archive_a[3] & 0x08, 0)  # FNAME flag must be unset.

            asset_a = (out_a / "asset_grouping_report.csv.gz").read_bytes()
            asset_b = (out_b / "asset_grouping_report.csv.gz").read_bytes()
            self.assertEqual(asset_a, asset_b)
            self.assertEqual(int.from_bytes(asset_a[4:8], "little"), 0)
            self.assertEqual(asset_a[3] & 0x08, 0)  # FNAME flag must be unset.

            hotspots_a = (out_a / "anomaly_hotspots.csv").read_text(encoding="utf-8")
            hotspots_b = (out_b / "anomaly_hotspots.csv").read_text(encoding="utf-8")
            self.assertEqual(hotspots_a, hotspots_b)

            summary_a = json.loads((out_a / "summary.json").read_text(encoding="utf-8"))
            summary_b = json.loads((out_b / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_a, summary_b)

            asset_rows = self._read_csv_gz_rows(out_a / "asset_grouping_report.csv.gz")
            orphan_group = next(row for row in asset_rows if row["basekey"].endswith("/ORPHAN"))
            clip_group = next(row for row in asset_rows if row["basekey"].endswith("/CLIP001"))
            self.assertEqual(orphan_group["flag_xmp_orphan_no_raw_jpeg"], "True")
            self.assertEqual(orphan_group["flag_xmp_orphan_no_image_any_raster"], "True")
            self.assertEqual(orphan_group["flag_sidecar_only"], "True")
            self.assertEqual(clip_group["flag_video_still_collision"], "True")

            with (out_a / "anomaly_hotspots.csv").open("r", encoding="utf-8", newline="") as handle:
                hotspot_rows = list(csv.DictReader(handle))
            self.assertIn("approx_bytes_decimal", hotspot_rows[0])
            self.assertGreater(int(float(hotspot_rows[0]["approx_bytes_decimal"])), 0)

    def test_cross_drive_basekey_collision_is_prevented(self) -> None:
        """Verify that identical basenames on different drives are treated as separate assets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "/vault/All Archive/DriveA/IMG_0001.CR2",
                        "FileName": "IMG_0001.CR2",
                        "FileSize": "10 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                    {
                        "SourceFile": "/vault/All Archive/DriveB/IMG_0001.CR2",
                        "FileName": "IMG_0001.CR2",
                        "FileSize": "12 MB",
                        "Model": "Nikon",
                        "DateTimeOriginal": "2024:02:01 14:00:00",
                        "ImageSize": "200x200",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                ],
            )

            result = self._run_cli(manifest_csv, outdir)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            asset_rows = self._read_csv_gz_rows(outdir / "asset_grouping_report.csv.gz")
            # Both should exist as separate asset groups
            drive_a_asset = next(
                (row for row in asset_rows if row["origin_drive"] == "DriveA" and row["basename"] == "IMG_0001"), None
            )
            drive_b_asset = next(
                (row for row in asset_rows if row["origin_drive"] == "DriveB" and row["basename"] == "IMG_0001"), None
            )

            self.assertIsNotNone(drive_a_asset, "DriveA asset with basename IMG_0001 should exist")
            self.assertIsNotNone(drive_b_asset, "DriveB asset with basename IMG_0001 should exist")
            self.assertEqual(drive_a_asset["n_raw_files"], "1")
            self.assertEqual(drive_b_asset["n_raw_files"], "1")
            # Verify basekeys are different (includes drive prefix for root-level files)
            self.assertNotEqual(drive_a_asset["basekey"], drive_b_asset["basekey"])

    def test_by_drive_outputs_avoid_filename_collisions_and_handle_root_placeholder(self) -> None:
        """Verify per-drive outputs are unique even when drive labels sanitize/truncate to same stem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            collision_stem = "D" * 80
            drive_a = collision_stem + "A" * 8
            drive_b = collision_stem + "B" * 8

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": f"/vault/All Archive/{drive_a}/Part1/IMG_COLLIDE_A.CR2",
                        "FileName": "IMG_COLLIDE_A.CR2",
                        "FileSize": "10 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                    {
                        "SourceFile": f"/vault/All Archive/{drive_b}/Part1/IMG_COLLIDE_B.CR2",
                        "FileName": "IMG_COLLIDE_B.CR2",
                        "FileSize": "12 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:02 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                    {
                        "SourceFile": "IMG_ROOT.CR2",
                        "FileName": "IMG_ROOT.CR2",
                        "FileSize": "8 MB",
                        "Model": "Nikon",
                        "DateTimeOriginal": "2024:01:03 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--input",
                    str(manifest_csv),
                    "--outdir",
                    str(outdir),
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            by_drive_dir = outdir / "by_origin_drive"
            self.assertTrue(by_drive_dir.exists())

            archive_files = sorted(by_drive_dir.glob("archive_index__*.csv.gz"))
            asset_files = sorted(by_drive_dir.glob("asset_groups__*.csv.gz"))
            self.assertEqual(len(archive_files), 3)
            self.assertEqual(len(asset_files), 3)
            self.assertEqual(len({p.name for p in archive_files}), 3)
            self.assertEqual(len({p.name for p in asset_files}), 3)

            self.assertTrue((by_drive_dir / "archive_index____ROOT__.csv.gz").exists())
            self.assertTrue((by_drive_dir / "asset_groups____ROOT__.csv.gz").exists())

            found_drives: set[str] = set()
            for path in archive_files:
                rows = self._read_csv_gz_rows(path)
                self.assertGreaterEqual(len(rows), 1)
                found_drives.update(row["origin_drive"] for row in rows)

            self.assertEqual(found_drives, {"", drive_a, drive_b})

            collision_labels = [
                path.name[len("archive_index__") : -len(".csv.gz")] for path in archive_files if "__ROOT__" not in path.name
            ]
            self.assertEqual(len(collision_labels), 2)
            self.assertTrue(any("__" in label for label in collision_labels))

    def test_absolute_paths_without_root_marker_get_leading_slash_stripped(self) -> None:
        """Verify that absolute paths without matching root_marker get leading slashes stripped.

        This prevents the bug where:
        - SourceFile="/vault/All Archive/DriveA/Part1/file.CR2"
        - root_marker not found → relpath uses full sf
        - leading "/" → dir_rel starts with "/" → parts[0] becomes ""
        - Result: origin_drive = "" (empty), partition = "vault" (WRONG!)

        After fix with lstrip("/"), we get:
        - relpath = "vault/All Archive/DriveA/Part1/file.CR2"
        - origin_drive = "vault", partition = "All Archive" (stable, deterministic)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            # Use absolute paths with a root_marker that won't match
            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "/vault/DriveA/Part1/IMG_0001.CR2",
                        "FileName": "IMG_0001.CR2",
                        "FileSize": "10 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                    {
                        "SourceFile": "/Volumes/RAID/Archive/DriveB/IMG_0002.JPG",
                        "FileName": "IMG_0002.JPG",
                        "FileSize": "2 MB",
                        "Model": "Nikon",
                        "DateTimeOriginal": "2024:02:01 14:00:00",
                        "ImageSize": "200x200",
                        "Quality": "",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                ],
            )

            # Use a root_marker that won't match (intentionally)
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--input",
                    str(manifest_csv),
                    "--outdir",
                    str(outdir),
                    "--root-marker",
                    "NONEXISTENT_MARKER/",
                    "--no-by-drive",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

            # Should succeed (with warning about marker coverage, but not fail)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            archive_rows = self._read_csv_gz_rows(outdir / "archive_index_normalized.csv.gz")

            # Verify that origin_drive is NOT empty (would be "" before fix)
            vault_row = next(row for row in archive_rows if "DriveA" in row["relpath"])
            volumes_row = next(row for row in archive_rows if "DriveB" in row["relpath"])

            # After lstrip("/"), first segment becomes "vault" and "Volumes" respectively
            self.assertEqual(vault_row["origin_drive"], "vault")
            self.assertEqual(vault_row["partition"], "DriveA")
            self.assertNotEqual(vault_row["origin_drive"], "")  # Critical: NOT empty

            self.assertEqual(volumes_row["origin_drive"], "Volumes")
            self.assertEqual(volumes_row["partition"], "RAID")
            self.assertNotEqual(volumes_row["origin_drive"], "")  # Critical: NOT empty

            # Verify relpath doesn't start with "/"
            self.assertFalse(vault_row["relpath"].startswith("/"))
            self.assertFalse(volumes_row["relpath"].startswith("/"))

    def test_strict_root_marker_fails_when_coverage_below_threshold(self) -> None:
        """Verify strict root marker mode fails closed when marker coverage is too low."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "/vault/All Archive/DriveA/IMG_0001.CR2",
                        "FileName": "IMG_0001.CR2",
                        "FileSize": "10 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                    {
                        "SourceFile": "/unmatched_path/DriveB/IMG_0002.JPG",
                        "FileName": "IMG_0002.JPG",
                        "FileSize": "2 MB",
                        "Model": "Nikon",
                        "DateTimeOriginal": "2024:01:01 10:01:00",
                        "ImageSize": "100x100",
                        "Quality": "",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                ],
            )

            result = self._run_cli(
                manifest_csv,
                outdir,
                extra_args=["--strict-root-marker", "--min-root-marker-coverage", "0.95"],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("root_marker coverage", result.stderr)

    def test_validate_schemas_rejects_invalid_asset_type_enum(self) -> None:
        """Verify schema validation catches enum-domain drift for governed categorical fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "/vault/All Archive/DriveA/IMG_0001.CR2",
                        "FileName": "IMG_0001.CR2",
                        "FileSize": "10 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    }
                ],
            )

            result = self._run_cli(manifest_csv, outdir)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            asset_path = outdir / "asset_grouping_report.csv.gz"
            asset_rows = self._read_csv_gz_rows(asset_path)
            self.assertGreaterEqual(len(asset_rows), 1)
            asset_rows[0]["asset_type"] = "invalid_asset_type"
            with gzip.open(asset_path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asset_rows[0].keys()))
                writer.writeheader()
                writer.writerows(asset_rows)

            tool_module = self._load_tool_module()
            with self.assertRaises(SystemExit) as ctx:
                tool_module.validate_outputs_against_schemas(outdir)

            msg = str(ctx.exception)
            self.assertIn("Schema validation failed for asset_grouping_report.csv.gz", msg)
            self.assertIn("invalid_asset_type", msg)

    def test_validate_schemas_rejects_invalid_summary_payload(self) -> None:
        """Verify summary.json is validated against schema when schema checks run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "/vault/All Archive/DriveA/IMG_0001.CR2",
                        "FileName": "IMG_0001.CR2",
                        "FileSize": "10 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    }
                ],
            )

            result = self._run_cli(manifest_csv, outdir)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            summary_path = outdir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["risk_weights"] = "not-an-object"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

            tool_module = self._load_tool_module()
            with self.assertRaises(SystemExit) as ctx:
                tool_module.validate_outputs_against_schemas(outdir)

            msg = str(ctx.exception)
            self.assertIn("Schema validation failed for summary.schema.json", msg)
            self.assertIn("risk_weights", msg)

    def test_unc_path_normalization_produces_stable_origin_drive(self) -> None:
        """Verify that Windows UNC paths (\\server\share\...) get normalized consistently.

        UNC paths convert:
        - \\server\share\DriveA\file.CR2
        - → //server/share/DriveA/file.CR2 (backslash→forward)
        - → /server/share/DriveA/file.CR2 (collapse //)
        - → server/share/DriveA/file.CR2 (lstrip "/")

        Result: origin_drive="server", partition="share" (stable, deterministic)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "\\\\fileserver\\archive_vault\\DriveA\\Part1\\IMG_0001.CR2",
                        "FileName": "IMG_0001.CR2",
                        "FileSize": "10 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--input",
                    str(manifest_csv),
                    "--outdir",
                    str(outdir),
                    "--root-marker",
                    "NONEXISTENT/",
                    "--no-by-drive",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            archive_rows = self._read_csv_gz_rows(outdir / "archive_index_normalized.csv.gz")
            unc_row = archive_rows[0]

            # Verify stable UNC parsing: origin_drive should be "fileserver"
            self.assertEqual(unc_row["origin_drive"], "fileserver")
            self.assertEqual(unc_row["partition"], "archive_vault")
            self.assertNotEqual(unc_row["origin_drive"], "")  # Not empty

            # Verify relpath is stable and doesn't start with "/"
            self.assertFalse(unc_row["relpath"].startswith("/"))
            self.assertIn("DriveA", unc_row["relpath"])

    def test_edge_case_root_slash_only_gets_placeholder(self) -> None:
        """Verify that edge-case paths like "/" get converted to placeholder "." to prevent empty strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "/",
                        "FileName": "",
                        "FileSize": "0 bytes",
                        "Model": "",
                        "DateTimeOriginal": "",
                        "ImageSize": "",
                        "Quality": "",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--input",
                    str(manifest_csv),
                    "--outdir",
                    str(outdir),
                    "--no-by-drive",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            archive_rows = self._read_csv_gz_rows(outdir / "archive_index_normalized.csv.gz")
            edge_row = archive_rows[0]

            # After lstrip("/"), we get "" which should be replaced with "."
            self.assertEqual(edge_row["relpath"], ".")
            # dir_rel will be empty, so origin_drive and partition should be ""
            self.assertEqual(edge_row["origin_drive"], "")
            self.assertEqual(edge_row["partition"], "")

    def test_multiple_occurrences_of_root_marker_uses_first_only(self) -> None:
        """Verify that if root_marker appears multiple times in path, only first occurrence is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            outdir = temp / "out"

            self._write_manifest(
                manifest_csv,
                [
                    {
                        "SourceFile": "/vault/All Archive/DriveA/All Archive/nested/IMG_0001.CR2",
                        "FileName": "IMG_0001.CR2",
                        "FileSize": "10 MB",
                        "Model": "Canon",
                        "DateTimeOriginal": "2024:01:01 10:00:00",
                        "ImageSize": "100x100",
                        "Quality": "RAW",
                        "FocalLength": "",
                        "ShutterSpeed": "",
                        "Aperture": "",
                        "ISO": "",
                        "WhiteBalance": "",
                        "Flash": "",
                    },
                ],
            )

            result = self._run_cli(manifest_csv, outdir)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            archive_rows = self._read_csv_gz_rows(outdir / "archive_index_normalized.csv.gz")
            multi_marker_row = archive_rows[0]

            # Should split on first "All Archive/" occurrence only (n=1 in split)
            # Result: relpath = "DriveA/All Archive/nested/IMG_0001.CR2"
            self.assertIn("DriveA", multi_marker_row["relpath"])
            self.assertIn("All Archive/nested", multi_marker_row["relpath"])
            self.assertEqual(multi_marker_row["origin_drive"], "DriveA")


if __name__ == "__main__":
    unittest.main()
