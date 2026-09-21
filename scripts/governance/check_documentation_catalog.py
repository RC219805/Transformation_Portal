#!/usr/bin/env python3
"""Validate the current documentation catalog without executing document text.

The catalog is a reviewed snapshot, not an automatic authority classifier.
Refresh only explicitly reviewed paths; never infer review status from age,
filename, a successful link check, or a passing test count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = "docs/governance/documentation_catalog.json"
SCHEMA = "tp.documentation.catalog.v1"
CLASSIFICATIONS = {"canonical", "current-support", "mixed", "historical", "archive-only"}
MAINTAINED = {"canonical", "current-support"}
REVIEW_STATUSES = {"source-reviewed", "inherited-classification", "historical-evidence", "generated-snapshot"}
EVIDENCE_TIERS = {"source-contract", "inventory-only", "historical-record", "generated-metadata"}
DOCUMENT_SUFFIXES = {".md", ".mdx", ".rst"}
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>\n]+>|(?:\\.|[^\s)])+)(?:\s+[\"'][^\n]*?[\"'])?\)")
REFERENCE_DEFINITION = re.compile(r"^ {0,3}\[([^\]]+)\]:\s*(<[^>]+>|\S+)")
REFERENCE_LINK = re.compile(r"!?\[([^\]\n]+)\]\[([^\]\n]*)\]")
HTML_LINK = re.compile(r"<(?:a|img)\b[^>]*\b(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
HISTORICAL_QUALIFIER = re.compile(
    r"\b(?:historic(?:al)?|history|dated|prior|audit|inventory|evidence|snapshot|backlog)\b|\b20\d{2}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def document_inventory(root: Path) -> set[str]:
    """Include tracked and prospective non-ignored documents, without walking deps."""
    paths = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    return {
        path
        for path in paths
        if path and (path.startswith("docs/") or Path(path).suffix.lower() in DOCUMENT_SUFFIXES) and (root / path).exists()
    }


def safe_path(root: Path, value: object) -> Path:
    """Reject aliases, traversal and symlinks in authority-bearing path fields."""
    if not isinstance(value, str) or not value or "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError(f"invalid repository path: {value!r}")
    relative = PurePosixPath(value)
    if (
        not relative.parts
        or relative.is_absolute()
        or str(relative) != value
        or any(part in {".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"non-canonical repository path: {value!r}")
    path = root / value
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root and root in parent.parents):
        raise ValueError(f"symlink is not documentation authority: {value}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"repository path escapes root: {value}")
    return path


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_catalog(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"expected catalog schema {SCHEMA}")
    return payload


def prose_lines(text: str) -> list[tuple[int, str]]:
    """Keep prose and line numbers; exclude fenced, indented and inline code."""
    result = []
    fence: tuple[str, int] | None = None
    in_comment = False
    for number, line in enumerate(text.splitlines(), 1):
        marker = FENCE.match(line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1]:
                fence = None
            continue
        if marker:
            fence = (marker[1][0], len(marker[1]))
            continue
        if line.startswith(("    ", "\t")):
            continue
        if in_comment:
            if "-->" not in line:
                continue
            line = line.split("-->", 1)[1]
            in_comment = False
        while "<!--" in line:
            before, after = line.split("<!--", 1)
            if "-->" in after:
                line = before + after.split("-->", 1)[1]
            else:
                line = before
                in_comment = True
        line = re.sub(r"(`+).*?\1", "", line)
        result.append((number, line))
    return result


def local_links(text: str) -> list[tuple[int, str]]:
    """Extract inline, explicit reference, shortcut reference and HTML links."""
    lines = prose_lines(text)
    definitions = {}
    for _number, line in lines:
        match = REFERENCE_DEFINITION.match(line)
        if match:
            definitions[" ".join(match[1].lower().split())] = match[2].strip("<>")
    links = []
    for number, line in lines:
        if REFERENCE_DEFINITION.match(line):
            continue
        found = [match[1].strip("<>") for match in INLINE_LINK.finditer(line)]
        for match in REFERENCE_LINK.finditer(line):
            label = " ".join((match[2] or match[1]).lower().split())
            if label in definitions:
                found.append(definitions[label])
        remaining = INLINE_LINK.sub("", REFERENCE_LINK.sub("", line))
        for label in re.findall(r"(?<!!)\[([^\]\n]+)\]", remaining):
            normalized = " ".join(label.lower().split())
            if normalized in definitions:
                found.append(definitions[normalized])
        found.extend(match[1] for match in HTML_LINK.finditer(line))
        for href in found:
            parsed = urlsplit(href)
            if not parsed.scheme and not parsed.netloc:
                links.append((number, href))
    return links


def _visible_navigation_context(text: str) -> dict[int, str]:
    """Exclude link destinations and HTML attributes from visible row/prose labels."""
    contexts = {}
    for number, line in prose_lines(text):
        visible = INLINE_LINK.sub(lambda match: match[0].split("](", 1)[0] + "]", line)
        visible = re.sub(r"<[^>]*>", "", visible)
        contexts[number] = visible.replace("_", " ")
    return contexts


def _resolve_link(root: Path, source: Path, href: str) -> tuple[Path, str]:
    parsed = urlsplit(href)
    raw = unquote(parsed.path)
    raw = re.sub(r"\\([()])", r"\1", raw)
    target = source if not raw else root / raw.lstrip("/") if raw.startswith("/") else source.parent / raw
    resolved = target.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("link escapes repository")
    return resolved, unquote(parsed.fragment)


def _anchors(path: Path) -> set[str]:
    """GitHub-style ATX/setext headings plus explicit HTML anchors."""
    text = path.read_text(encoding="utf-8")
    lines = prose_lines(text)
    prose = "\n".join(line for _number, line in lines)
    anchors = set(re.findall(r"\b(?:id|name)=[\"']([^\"']+)[\"']", prose))
    counts: dict[str, int] = {}
    previous = ""
    for _number, line in lines:
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        heading = match[1] if match else previous if re.match(r"^ {0,3}(?:=+|-+)\s*$", line) else None
        previous = line.strip()
        if heading is None:
            continue
        # Inline code is meaningful heading text; restore from the original line.
        original = text.splitlines()[_number - 1]
        if match:
            heading = re.sub(r"^ {0,3}#{1,6}\s+|\s+#+\s*$", "", original)
        heading = re.sub(r"<[^>]*>", "", heading).lower().strip()
        slug = re.sub(r"[^\w\s-]", "", heading)
        slug = re.sub(r"\s", "-", slug)
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def validate_catalog(root: Path, catalog: dict, *, inventory: set[str] | None = None, check_links: bool = True) -> list[str]:
    """Check closure, review provenance, graph and maintained navigation contracts."""
    failures = []
    if catalog.get("schema") != SCHEMA:
        return [f"expected schema {SCHEMA}"]
    if not isinstance(catalog.get("source_baseline"), str) or not re.fullmatch(r"[0-9a-f]{40}", catalog["source_baseline"]):
        failures.append("source_baseline must name a 40-character source commit")
    records = catalog.get("documents")
    if not isinstance(records, list):
        return failures + ["documents must be an array"]
    entries = {}
    for record in records:
        if not isinstance(record, dict):
            failures.append("document record must be an object")
            continue
        name = record.get("path")
        try:
            path = safe_path(root, name)
        except ValueError as error:
            failures.append(str(error))
            continue
        if name in entries:
            failures.append(f"duplicate document: {name}")
        entries[name] = record
        if not path.is_file():
            failures.append(f"document is missing: {name}")
        if not isinstance(record.get("classification"), str) or record["classification"] not in CLASSIFICATIONS:
            failures.append(f"{name}: unknown classification")
        if not isinstance(record.get("review_status"), str) or record["review_status"] not in REVIEW_STATUSES:
            failures.append(f"{name}: unknown review_status")
        if not isinstance(record.get("evidence_tier"), str) or record["evidence_tier"] not in EVIDENCE_TIERS:
            failures.append(f"{name}: unknown evidence_tier")
        for field in ("scope", "maintenance_area"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                failures.append(f"{name}: {field} must be explicit")
        if not isinstance(record.get("source_commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", record["source_commit"]):
            failures.append(f"{name}: source_commit must identify reviewed source baseline")
        digest = record.get("content_sha256")
        if name == CATALOG_PATH:
            if digest is not None or record.get("review_status") != "generated-snapshot":
                failures.append("catalog self-record must be unhashed generated-snapshot")
        elif not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            failures.append(f"{name}: content_sha256 must bind document bytes")
        elif path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(f"{name}: content changed; review and explicitly refresh this entry")
        for field in ("successors", "source_references", "validation_commands"):
            values = record.get(field)
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                failures.append(f"{name}: {field} must be an array of nonempty strings")
        if record.get("review_status") == "source-reviewed":
            if record.get("evidence_tier") != "source-contract" or not record.get("source_references"):
                failures.append(f"{name}: source-reviewed requires source-contract evidence and references")
            source_hashes = record.get("source_sha256")
            references = record.get("source_references")
            if (
                not isinstance(source_hashes, dict)
                or not isinstance(references, list)
                or any(not isinstance(reference, str) for reference in references)
                or set(source_hashes) != set(references)
            ):
                failures.append(f"{name}: source-reviewed requires a hash for each source reference")
        for reference in record.get("source_references", []) if isinstance(record.get("source_references"), list) else []:
            try:
                source_path = safe_path(root, reference)
                if reference == CATALOG_PATH:
                    failures.append(f"{name}: catalog cannot hash itself through a source reference")
                elif not source_path.is_file():
                    failures.append(f"{name}: missing source reference: {reference}")
                elif record.get("review_status") == "source-reviewed" and isinstance(record.get("source_sha256"), dict):
                    if record["source_sha256"].get(reference) != hashlib.sha256(source_path.read_bytes()).hexdigest():
                        failures.append(f"{name}: reviewed source changed: {reference}")
            except ValueError as error:
                failures.append(f"{name}: {error}")
    expected = document_inventory(root) if inventory is None else inventory
    for name in sorted(expected - entries.keys()):
        failures.append(f"uncataloged document: {name}")
    for name in sorted(entries.keys() - expected):
        failures.append(f"catalog entry outside current inventory: {name}")
    if list(entries) != sorted(entries):
        failures.append("documents must be sorted by path")

    visiting, visited = set(), set()

    def visit(name: str) -> None:
        if name in visiting:
            failures.append(f"successor cycle at {name}")
            return
        if name in visited:
            return
        visiting.add(name)
        successors = entries[name].get("successors", [])
        for successor in successors if isinstance(successors, list) else []:
            if not isinstance(successor, str) or successor not in entries:
                failures.append(f"{name}: unknown successor: {successor!r}")
            else:
                visit(successor)
        visiting.remove(name)
        visited.add(name)

    for name in entries:
        visit(name)
    if check_links:
        failures.extend(validate_links(root, catalog, entries))
    return failures


def validate_links(root: Path, catalog: dict, entries: dict) -> list[str]:
    """Validate maintained local links and explicit historical navigation labels."""
    failures = []
    navigation = catalog.get("navigation_sources", [])
    exceptions = catalog.get("historical_navigation", [])
    if not isinstance(navigation, list) or any(not isinstance(value, str) or value not in entries for value in navigation):
        return ["navigation_sources must contain cataloged document paths"]
    allowed = set()
    if not isinstance(exceptions, list):
        return ["historical_navigation must be an array"]
    for exception in exceptions:
        if (
            not isinstance(exception, dict)
            or not isinstance(exception.get("source"), str)
            or not isinstance(exception.get("target"), str)
            or exception.get("source") not in navigation
            or exception.get("target") not in entries
            or not isinstance(exception.get("reason"), str)
            or not exception["reason"].strip()
        ):
            failures.append("historical navigation exception must name valid source, target and reason")
            continue
        allowed.add((exception["source"], exception["target"]))
    observed = set()
    anchor_cache = {}
    for name, record in entries.items():
        if not isinstance(record.get("classification"), str) or record["classification"] not in MAINTAINED:
            continue
        if Path(name).suffix.lower() not in {".md", ".mdx"}:
            continue
        try:
            source = safe_path(root, name)
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError) as error:
            failures.append(f"{name}: cannot read maintained document for link validation: {error}")
            continue
        visible_context = _visible_navigation_context(text) if name in navigation else {}
        for line, href in local_links(text):
            try:
                target, anchor = _resolve_link(root, source, href)
            except ValueError as error:
                failures.append(f"{name}:{line}: {error}: {href}")
                continue
            if not target.exists():
                failures.append(f"{name}:{line}: missing local link: {href}")
                continue
            if anchor and target.is_file() and target.suffix.lower() in {".md", ".mdx"}:
                if target not in anchor_cache:
                    anchor_cache[target] = _anchors(target)
                if anchor not in anchor_cache[target]:
                    failures.append(f"{name}:{line}: missing heading #{anchor}: {href}")
            relative = target.relative_to(root.resolve()).as_posix()
            pair = (name, relative)
            if name in navigation and relative in entries and entries[relative].get("classification") not in MAINTAINED:
                observed.add(pair)
                if pair not in allowed:
                    failures.append(
                        f"{name}:{line}: navigation promotes non-maintained {relative}; label historical evidence explicitly"
                    )
                elif not HISTORICAL_QUALIFIER.search(visible_context.get(line, "")):
                    failures.append(
                        f"{name}:{line}: historical navigation requires a visible dated/evidence qualifier: {relative}"
                    )
    for pair in sorted(allowed - observed):
        failures.append(f"stale historical navigation exception: {pair[0]} -> {pair[1]}")
    return failures


def refresh_entries(root: Path, catalog: dict, paths: list[str], source_commit: str) -> None:
    """Refresh explicitly reviewed bytes; never infer or elevate classification."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("--source-commit requires the exact reviewed source baseline SHA")
    entries = {entry["path"]: entry for entry in catalog["documents"]}
    for name in paths:
        if name not in entries or name == CATALOG_PATH:
            raise ValueError(f"add/classify the document explicitly before refresh: {name}")
        path = safe_path(root, name)
        entries[name]["content_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        entries[name]["source_commit"] = source_commit
        if entries[name].get("review_status") == "source-reviewed":
            entries[name]["source_sha256"] = {
                reference: hashlib.sha256(safe_path(root, reference).read_bytes()).hexdigest()
                for reference in entries[name]["source_references"]
            }


def write_catalog(path: Path, catalog: dict) -> None:
    """Publish a complete validated snapshot atomically in its own directory."""
    encoded = (json.dumps(catalog, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".documentation-catalog-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--refresh", nargs="+", metavar="PATH", help="Refresh only explicitly reviewed, already cataloged files"
    )
    parser.add_argument("--source-commit", help="Exact reviewed source baseline, required with --refresh")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        path = safe_path(root, CATALOG_PATH)
        catalog = load_catalog(path)
        if args.refresh:
            refresh_entries(root, catalog, args.refresh, args.source_commit or "")
        failures = validate_catalog(root, catalog)
        if failures:
            for failure in failures:
                print(f"ERROR: {failure}")
            return 1
        if args.refresh:
            write_catalog(path, catalog)
        print(
            f"Documentation catalog passed ({len(catalog['documents'])} documents; maintained links and navigation checked)."
        )
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: documentation catalog validation unavailable: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
