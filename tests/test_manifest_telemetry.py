"""CLI tests for tools/manifest_telemetry.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

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

    def test_merkle_proof_generates_valid_proof(self) -> None:
        """Verify merkle-proof generates a proof that verifies correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            proof_json = temp / "proof.json"
            verify_json = temp / "verify.json"

            with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "bytes", "md5"])
                writer.writerow(["driveA/file1.txt", "1", "aaa"])
                writer.writerow(["driveA/file2.txt", "2", "bbb"])
                writer.writerow(["driveB/file3.txt", "3", "ccc"])

            # Generate proof for first file
            proof_result = self._run_cli(
                "merkle-proof",
                "--manifest",
                str(manifest_csv),
                "--filename",
                "driveA/file1.txt",
                "--out-json",
                str(proof_json),
            )
            self.assertEqual(proof_result.returncode, 0, msg=proof_result.stderr)
            self.assertTrue(proof_json.exists())

            proof_payload = json.loads(proof_json.read_text(encoding="utf-8"))
            self.assertEqual(proof_payload["schema_version"], "1.0")
            self.assertEqual(proof_payload["target"]["filename"], "driveA/file1.txt")
            self.assertIn("leaf_hex", proof_payload)
            self.assertIn("root_hex", proof_payload)
            self.assertIn("proof", proof_payload)
            self.assertIn("contract", proof_payload)

            # Verify the proof
            verify_result = self._run_cli(
                "merkle-verify",
                "--proof-json",
                str(proof_json),
                "--out-json",
                str(verify_json),
            )
            self.assertEqual(verify_result.returncode, 0, msg=verify_result.stderr)
            self.assertTrue(verify_json.exists())

            verify_payload = json.loads(verify_json.read_text(encoding="utf-8"))
            self.assertTrue(verify_payload["verified"])
            self.assertEqual(verify_payload["expected_root_hex"], verify_payload["computed_root_hex"])

    def test_merkle_proof_root_matches_merkle_command(self) -> None:
        """Verify merkle-proof generates a root identical to the merkle command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            merkle_json = temp / "merkle.json"
            proof_json = temp / "proof.json"

            with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "bytes", "md5"])
                writer.writerow(["driveA/file1.txt", "1", "aaa"])
                writer.writerow(["driveA/file2.txt", "2", "bbb"])

            # Generate merkle root
            merkle_result = self._run_cli(
                "merkle",
                "--manifest",
                str(manifest_csv),
                "--out-json",
                str(merkle_json),
            )
            self.assertEqual(merkle_result.returncode, 0, msg=merkle_result.stderr)

            # Generate proof for any file
            proof_result = self._run_cli(
                "merkle-proof",
                "--manifest",
                str(manifest_csv),
                "--filename",
                "driveA/file1.txt",
                "--out-json",
                str(proof_json),
            )
            self.assertEqual(proof_result.returncode, 0, msg=proof_result.stderr)

            merkle_payload = json.loads(merkle_json.read_text(encoding="utf-8"))
            proof_payload = json.loads(proof_json.read_text(encoding="utf-8"))

            # Root in proof must match global root from merkle command
            self.assertEqual(
                merkle_payload["global_root"],
                proof_payload["root_hex"],
                "Merkle proof root must match global root from merkle command",
            )

    def test_merkle_verify_fails_on_tampered_proof(self) -> None:
        """Verify merkle-verify fails (exit code 2) when proof is tampered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            proof_json = temp / "proof.json"
            verify_json = temp / "verify.json"

            with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "bytes", "md5"])
                writer.writerow(["driveA/file1.txt", "1", "aaa"])

            # Generate valid proof
            self._run_cli(
                "merkle-proof",
                "--manifest",
                str(manifest_csv),
                "--filename",
                "driveA/file1.txt",
                "--out-json",
                str(proof_json),
            )

            # Tamper with the proof
            proof_payload = json.loads(proof_json.read_text(encoding="utf-8"))
            proof_payload["root_hex"] = "badc0de" * 8
            proof_json.write_text(json.dumps(proof_payload), encoding="utf-8")

            # Verification should fail with exit code 2
            verify_result = self._run_cli(
                "merkle-verify",
                "--proof-json",
                str(proof_json),
                "--out-json",
                str(verify_json),
            )
            self.assertEqual(verify_result.returncode, 2)

            verify_payload = json.loads(verify_json.read_text(encoding="utf-8"))
            self.assertFalse(verify_payload["verified"])
            self.assertNotEqual(verify_payload["expected_root_hex"], verify_payload["computed_root_hex"])

    def _write_three_row_manifest(self, manifest_csv: Path, *, digest_suffix: str = "") -> None:
        with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "bytes", "md5"])
            writer.writerow(["driveA/a.jpg", "10", f"aaaa{digest_suffix}"])
            writer.writerow(["driveA/b.jpg", "20", f"bbbb{digest_suffix}"])
            writer.writerow(["driveB/c.jpg", "30", f"cccc{digest_suffix}"])

    def _merkle_and_proof(self, temp: Path, name: str, *, digest_suffix: str = "") -> tuple[Path, Path]:
        manifest_csv = temp / f"{name}.csv"
        merkle_json = temp / f"{name}.merkle.json"
        proof_json = temp / f"{name}.proof.json"
        self._write_three_row_manifest(manifest_csv, digest_suffix=digest_suffix)
        merkle = self._run_cli("merkle", "--manifest", str(manifest_csv), "--out-json", str(merkle_json))
        self.assertEqual(merkle.returncode, 0, msg=merkle.stderr)
        proof = self._run_cli(
            "merkle-proof", "--manifest", str(manifest_csv), "--filename", "driveA/a.jpg", "--out-json", str(proof_json)
        )
        self.assertEqual(proof.returncode, 0, msg=proof.stderr)
        return merkle_json, proof_json

    def test_merkle_verify_rejects_proof_whose_target_was_swapped(self) -> None:
        """The leaf must be the hash of the row the proof names, or the proof is about some other file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            _, proof_json = self._merkle_and_proof(temp, "manifest")
            verify_json = temp / "verify.json"

            proof_payload = json.loads(proof_json.read_text(encoding="utf-8"))
            proof_payload["target"] = {"filename": "driveZ/never_existed.jpg", "bytes": 999, "digest": "ffff", "leaf_index": 0}
            proof_json.write_text(json.dumps(proof_payload), encoding="utf-8")

            result = self._run_cli("merkle-verify", "--proof-json", str(proof_json), "--out-json", str(verify_json))

            self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
            report = json.loads(verify_json.read_text(encoding="utf-8"))
            self.assertFalse(report["verified"])
            self.assertEqual(report["failures"], ["leaf_not_bound_to_target"])

    def test_merkle_verify_rejects_fabricated_proof_without_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            proof_json = temp / "proof.json"
            verify_json = temp / "verify.json"
            proof_json.write_text(json.dumps({"leaf_hex": "00" * 32, "root_hex": "00" * 32, "proof": []}), encoding="utf-8")

            result = self._run_cli("merkle-verify", "--proof-json", str(proof_json), "--out-json", str(verify_json))

            self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
            self.assertFalse(json.loads(verify_json.read_text(encoding="utf-8"))["verified"])

    def test_merkle_verify_without_anchor_is_reported_as_unanchored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            _, proof_json = self._merkle_and_proof(temp, "manifest")
            verify_json = temp / "verify.json"

            result = self._run_cli("merkle-verify", "--proof-json", str(proof_json), "--out-json", str(verify_json))

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("not compared with a trusted Merkle root", result.stderr)
            report = json.loads(verify_json.read_text(encoding="utf-8"))
            self.assertTrue(report["verified"])
            self.assertFalse(report["anchored"])

    def test_merkle_verify_anchors_proof_to_trusted_merkle_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            trusted_merkle_json, genuine_proof = self._merkle_and_proof(temp, "trusted")
            _, foreign_proof = self._merkle_and_proof(temp, "foreign", digest_suffix="-edited")
            verify_json = temp / "verify.json"

            genuine = self._run_cli(
                "merkle-verify",
                "--proof-json",
                str(genuine_proof),
                "--out-json",
                str(verify_json),
                "--merkle-json",
                str(trusted_merkle_json),
            )
            self.assertEqual(genuine.returncode, 0, msg=genuine.stdout + genuine.stderr)
            self.assertEqual(genuine.stderr, "")
            report = json.loads(verify_json.read_text(encoding="utf-8"))
            self.assertTrue(report["verified"])
            self.assertTrue(report["anchored"])
            self.assertEqual(report["failures"], [])

            # Internally consistent, but it proves membership in a different manifest.
            foreign = self._run_cli(
                "merkle-verify",
                "--proof-json",
                str(foreign_proof),
                "--out-json",
                str(verify_json),
                "--merkle-json",
                str(trusted_merkle_json),
            )
            self.assertEqual(foreign.returncode, 2, msg=foreign.stdout + foreign.stderr)
            self.assertEqual(json.loads(verify_json.read_text(encoding="utf-8"))["failures"], ["root_not_trusted"])

    def test_merkle_verify_expected_root_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            trusted_merkle_json, proof_json = self._merkle_and_proof(temp, "manifest")
            verify_json = temp / "verify.json"
            global_root = json.loads(trusted_merkle_json.read_text(encoding="utf-8"))["global_root"]

            matching = self._run_cli(
                "merkle-verify",
                "--proof-json",
                str(proof_json),
                "--out-json",
                str(verify_json),
                "--expected-root",
                global_root.upper(),
            )
            self.assertEqual(matching.returncode, 0, msg=matching.stdout + matching.stderr)
            self.assertTrue(json.loads(verify_json.read_text(encoding="utf-8"))["anchored"])

            other = self._run_cli(
                "merkle-verify",
                "--proof-json",
                str(proof_json),
                "--out-json",
                str(verify_json),
                "--expected-root",
                "0" * 64,
            )
            self.assertEqual(other.returncode, 2, msg=other.stdout + other.stderr)
            self.assertEqual(json.loads(verify_json.read_text(encoding="utf-8"))["failures"], ["root_not_trusted"])

    def test_merkle_verify_rejects_single_node_proof_for_multi_leaf_tree(self) -> None:
        """An empty proof says "this leaf is the root"; that is only possible in a one-leaf tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            trusted_merkle_json, proof_json = self._merkle_and_proof(temp, "manifest")
            verify_json = temp / "verify.json"

            proof_payload = json.loads(proof_json.read_text(encoding="utf-8"))
            proof_payload["proof"] = []
            proof_payload["root_hex"] = proof_payload["leaf_hex"]
            proof_json.write_text(json.dumps(proof_payload), encoding="utf-8")

            result = self._run_cli(
                "merkle-verify",
                "--proof-json",
                str(proof_json),
                "--out-json",
                str(verify_json),
                "--merkle-json",
                str(trusted_merkle_json),
            )

            self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
            self.assertEqual(
                json.loads(verify_json.read_text(encoding="utf-8"))["failures"],
                ["proof_length_mismatch", "root_not_trusted"],
            )

    def test_merkle_proof_fails_on_missing_filename(self) -> None:
        """Verify merkle-proof fails gracefully when filename not in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            manifest_csv = temp / "manifest.csv"
            proof_json = temp / "proof.json"

            with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "bytes", "md5"])
                writer.writerow(["driveA/file1.txt", "1", "aaa"])

            result = self._run_cli(
                "merkle-proof",
                "--manifest",
                str(manifest_csv),
                "--filename",
                "nonexistent.txt",
                "--out-json",
                str(proof_json),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not found", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
