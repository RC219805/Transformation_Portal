"""Bind documentation retrieval authority to a reviewed catalog and source bytes."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

CATALOG_PATH = "docs/governance/documentation_catalog.json"
CATALOG_SCHEMA = "tp.documentation.catalog.v1"
CLASSIFICATIONS = {"canonical", "current-support", "mixed", "historical", "archive-only"}
REVIEW_STATUSES = {"source-reviewed", "inherited-classification", "historical-evidence", "generated-snapshot"}
EVIDENCE_TIERS = {"source-contract", "inventory-only", "historical-record", "generated-metadata"}
RETRIEVAL_MODES = ("operator", "historical", "all")


def _relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or any(ord(char) < 32 for char in value):
        return False
    path = PurePosixPath(value)
    return (
        bool(path.parts)
        and not path.is_absolute()
        and path.as_posix() == value
        and not any(part in {".", ".."} for part in path.parts)
    )


def _unique_object(pairs):
    """Reject ambiguous JSON members at every catalog object depth."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _has_successor_cycle(documents: Dict[str, Dict[str, Any]]) -> bool:
    """Check known successor closure iteratively, including self references."""
    visited, active = set(), set()
    for start in documents:
        stack = [(start, False)]
        while stack:
            name, leaving = stack.pop()
            if leaving:
                active.remove(name)
                visited.add(name)
                continue
            if name in active:
                return True
            if name in visited:
                continue
            active.add(name)
            stack.append((name, True))
            successors = documents[name].get("successors")
            if isinstance(successors, list):
                for successor in reversed(successors):
                    if isinstance(successor, str) and successor in documents:
                        stack.append((successor, False))
    return False


@dataclass
class DocumentationCatalog:
    """One immutable read snapshot; invalid evidence never grants maintained status."""

    repo_root: Path
    digest: Optional[str]
    status: str
    documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_baseline: Optional[str] = None
    source_hashes: Dict[str, Optional[str]] = field(default_factory=dict)

    @property
    def fingerprint(self) -> Optional[str]:
        """Bind the raw catalog and every consulted source snapshot, even outside the corpus."""
        if self.digest is None:
            return None
        payload = [self.digest, sorted(self.source_hashes.items())]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _safe_file(repo_root: Path, relative_path: str) -> bool:
        target = repo_root / relative_path
        return (
            target.is_file()
            and target.resolve().is_relative_to(repo_root.resolve())
            and not any(
                part.is_symlink() for part in (target, *target.parents) if part != repo_root and repo_root in part.parents
            )
        )

    @staticmethod
    def _source_hash(repo_root: Path, relative_path: str) -> Optional[str]:
        target = repo_root / relative_path
        try:
            if not DocumentationCatalog._safe_file(repo_root, relative_path):
                return None
            digest = hashlib.sha256()
            with target.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except (OSError, RuntimeError):
            return None

    @classmethod
    def read(cls, repo_root: Path) -> "DocumentationCatalog":
        path = repo_root / CATALOG_PATH
        try:
            if not cls._safe_file(repo_root, CATALOG_PATH):
                if not path.exists() and not path.is_symlink():
                    return cls(repo_root, "missing", "missing-catalog")
                return cls(repo_root, None, "unsafe-catalog-path")
            raw = path.read_bytes()
        except FileNotFoundError:
            return cls(repo_root, "missing", "missing-catalog")
        except (OSError, RuntimeError):
            return cls(repo_root, None, "unreadable-catalog")
        digest = hashlib.sha256(raw).hexdigest()
        try:
            payload = json.loads(raw, object_pairs_hook=_unique_object)
            if not isinstance(payload, dict) or payload.get("schema") != CATALOG_SCHEMA:
                raise ValueError("unsupported catalog schema")
            baseline = payload.get("source_baseline")
            if not isinstance(baseline, str) or not re.fullmatch(r"[0-9a-f]{40}", baseline):
                raise ValueError("invalid source baseline")
            entries = payload.get("documents")
            if not isinstance(entries, list):
                raise ValueError("invalid documents")
            documents = {}
            for entry in entries:
                if not isinstance(entry, dict) or not _relative_path(entry.get("path")):
                    raise ValueError("invalid document path")
                if entry["path"] in documents:
                    raise ValueError("duplicate document path")
                documents[entry["path"]] = entry
            if _has_successor_cycle(documents):
                raise ValueError("cyclic successor closure")
            source_paths = set()
            for entry in entries:
                references = entry.get("source_references")
                if isinstance(references, list):
                    source_paths.update(ref for ref in references if _relative_path(ref) and ref != CATALOG_PATH)
            source_hashes = {ref: cls._source_hash(repo_root, ref) for ref in sorted(source_paths)}
            return cls(repo_root, digest, "valid", documents, baseline, source_hashes)
        except (ValueError, UnicodeDecodeError):
            return cls(repo_root, digest, "invalid-catalog")

    def metadata(self, path: str, source_hash: str) -> Dict[str, Any]:
        """Classify the exact chunked bytes, retaining declared status as evidence."""
        result = {"authority": "unverified", "verification": self.status, "catalog_sha256": self.digest}
        if self.status != "valid":
            return result
        entry = self.documents.get(path)
        if entry is None:
            result["verification"] = "missing-entry"
            return result
        classification = entry.get("classification")
        digest = entry.get("content_sha256")
        successors = entry.get("successors")
        strings = ("review_status", "evidence_tier", "maintenance_area", "scope", "source_commit")
        if (
            not isinstance(classification, str)
            or classification not in CLASSIFICATIONS
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not all(isinstance(entry.get(key), str) and entry[key].strip() for key in strings)
            or entry["review_status"] not in REVIEW_STATUSES
            or entry["evidence_tier"] not in EVIDENCE_TIERS
            or not re.fullmatch(r"[0-9a-f]{40}", entry["source_commit"])
            or not isinstance(successors, list)
            or not all(_relative_path(successor) for successor in successors)
        ):
            result["verification"] = "invalid-entry"
            return result
        result.update({key: entry[key] for key in (*strings, "classification", "content_sha256", "successors")})
        result["source_baseline"] = self.source_baseline
        if not self._safe_file(self.repo_root, path):
            result["verification"] = "unsafe-document-path"
            return result
        if digest != source_hash:
            result["verification"] = "stale-content"
            return result
        for successor in successors:
            if successor not in self.documents or not self._safe_file(self.repo_root, successor):
                result["verification"] = "unresolved-successor"
                return result
        if entry["review_status"] == "source-reviewed":
            references = entry.get("source_references")
            expected = entry.get("source_sha256")
            if (
                entry["evidence_tier"] != "source-contract"
                or not isinstance(references, list)
                or not references
                or not all(_relative_path(ref) and ref != CATALOG_PATH for ref in references)
                or not isinstance(expected, dict)
                or set(expected) != set(references)
                or not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in expected.values())
            ):
                result["verification"] = "invalid-source-provenance"
                return result
            if any(self.source_hashes.get(ref) != expected[ref] for ref in references):
                result["verification"] = "stale-source"
                return result
            result["source_references"] = references
            result["source_sha256"] = expected
        result["verification"] = "verified"
        if classification in {"historical", "archive-only"}:
            result["authority"] = "historical"
        elif classification in {"canonical", "current-support"} and entry["review_status"] == "source-reviewed":
            result["authority"] = "maintained"
        else:
            result["verification"] = "unreviewed"
        return result


def authority_priority(metadata: Dict[str, Any]) -> int:
    """Only an explicitly verified maintained snapshot receives operator priority."""
    authority = metadata.get("documentation", {})
    return int(authority.get("authority") == "maintained" and authority.get("verification") == "verified")


def authority_note(metadata: Dict[str, Any]) -> str:
    """Human-readable provenance, independent of retrieval relevance scores."""
    authority = metadata.get("documentation")
    if not authority:
        return "Authority: unclassified source"
    note = f"Authority: {authority['authority']} ({authority['verification']})"
    if authority.get("classification"):
        note += f"; classification: {authority['classification']}"
    if authority.get("evidence_tier"):
        note += f"; evidence: {authority['evidence_tier']}"
    if authority.get("review_status"):
        note += f"; review: {authority['review_status']}"
    if authority.get("successors"):
        note += "; successors: " + ", ".join(authority["successors"])
    return note
