"""CLI tests for tools/manifest_telemetry.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "manifest_telemetry.py"


class ManifestTelemetryCliTest(unittest.TestCase):
    """Validate deterministic telemetry outputs and governance gate behavior."""

    def _run_cli(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, str(TOOL_PATH), *args]
        return subprocess.run(
            cmd,
            cwd=str(cwd or PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_metrics_writes_json_and_audit_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            out_json = temp / "metrics.json"
            out_audit = temp / "audit.csv"

            with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "bytes", "md5"])
                writer.writerow(["driveA/image_001.jpg", "10", "aa"])
                writer.writerow(["driveB/image_002.jpg", "20", "aa"])
                writer.writerow(["driveB/image_003.png", "30", "bb"])

            result = self._run_cli(
                "metrics",
                "--manifest",
                str(manifest_csv),
                "--out-json",
                str(out_json),
                "--out-audit-csv",
                str(out_audit),
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_audit.exists())

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["manifest_name"], "manifest.csv")
            self.assertEqual(payload["digest_column"], "md5")
            self.assertEqual(payload["digest_algorithm"], "md5")
            self.assertNotIn("manifest_path", payload)
            self.assertEqual(payload["totals"]["files"], 3)
            self.assertEqual(payload["totals"]["bytes"], 60)
            self.assertEqual(payload["duplicates"]["groups"], 1)
            self.assertEqual(payload["duplicates"]["excess_files"], 1)

            with out_audit.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["filename"], "driveA/image_001.jpg")
            self.assertEqual(rows[0]["digest"], "aa")
            self.assertIn("top_level_dir", rows[0])
            self.assertNotIn("md5", rows[0])

    def test_metrics_labels_sha256_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            out_json = temp / "metrics.json"
            out_audit = temp / "audit.csv"

            with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "bytes", "sha256"])
                writer.writerow(["driveA/image_001.jpg", "10", "abc123"])

            result = self._run_cli(
                "metrics",
                "--manifest",
                str(manifest_csv),
                "--out-json",
                str(out_json),
                "--out-audit-csv",
                str(out_audit),
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["digest_column"], "sha256")
            self.assertEqual(payload["digest_algorithm"], "sha256")

            with out_audit.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["digest"], "abc123")

    def test_merkle_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            out_json_a = temp / "merkle_a.json"
            out_json_b = temp / "merkle_b.json"

            with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "bytes", "md5"])
                writer.writerow(["driveA/file1.txt", "1", "aaa"])
                writer.writerow(["driveA/file2.txt", "2", "bbb"])
                writer.writerow(["driveB/file3.txt", "3", "ccc"])

            first = self._run_cli("merkle", "--manifest", str(manifest_csv), "--out-json", str(out_json_a))
            second = self._run_cli("merkle", "--manifest", "manifest.csv", "--out-json", "merkle_b.json", cwd=temp)
            self.assertEqual(first.returncode, 0, msg=first.stderr)
            self.assertEqual(second.returncode, 0, msg=second.stderr)

            payload_a = json.loads(out_json_a.read_text(encoding="utf-8"))
            payload_b = json.loads(out_json_b.read_text(encoding="utf-8"))
            self.assertEqual(payload_a, payload_b)
            self.assertEqual(payload_a["manifest_name"], "manifest.csv")
            self.assertNotIn("manifest_path", payload_a)
            self.assertEqual(payload_a["global_root"], payload_b["global_root"])
            self.assertEqual(payload_a["leaf_count"], 3)
            self.assertEqual({entry["drive"] for entry in payload_a["per_drive_roots"]}, {"driveA", "driveB"})

    def test_governance_gate_pass_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            governance_csv = temp / "rights_privacy_governance.csv"
            gate_pass = temp / "gate_pass.json"
            gate_fail = temp / "gate_fail.json"

            with governance_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["asset_id", "classification"])
                writer.writerow(["1", "approved"])
                writer.writerow(["2", ""])
                writer.writerow(["3", "restricted"])

            ok = self._run_cli(
                "governance-gate",
                "--governance-csv",
                str(governance_csv),
                "--min-classified",
                "2",
                "--out-json",
                str(gate_pass),
            )
            self.assertEqual(ok.returncode, 0, msg=ok.stderr)
            payload_ok = json.loads(gate_pass.read_text(encoding="utf-8"))
            self.assertTrue(payload_ok["passed"])

            failing = self._run_cli(
                "governance-gate",
                "--governance-csv",
                str(governance_csv),
                "--min-classified",
                "3",
                "--out-json",
                str(gate_fail),
            )
            self.assertNotEqual(failing.returncode, 0)
            payload_fail = json.loads(gate_fail.read_text(encoding="utf-8"))
            self.assertFalse(payload_fail["passed"])

    def test_governance_gate_includes_threshold_mode(self) -> None:
        """Verify threshold_mode is 'rows' to avoid ambiguity with percentage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            governance_csv = temp / "rights_privacy_governance.csv"
            gate_json = temp / "gate.json"

            with governance_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["asset_id", "classification"])
                writer.writerow(["1", "approved"])

            result = self._run_cli(
                "governance-gate",
                "--governance-csv",
                str(governance_csv),
                "--min-classified",
                "1",
                "--out-json",
                str(gate_json),
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(gate_json.read_text(encoding="utf-8"))

            # Phase 1.1 requirement: explicit threshold_mode to prevent "95% vs 95 rows" ambiguity
            self.assertIn("threshold_mode", payload, "governance JSON must include threshold_mode")
            self.assertEqual(payload["threshold_mode"], "rows")


if __name__ == "__main__":
    unittest.main()
