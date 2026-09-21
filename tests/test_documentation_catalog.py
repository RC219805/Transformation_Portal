"""Authority, freshness and maintained-navigation contracts for documentation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.governance import check_documentation_catalog as catalog_check

pytestmark = pytest.mark.unit
BASELINE = "a" * 40


def _record(root: Path, path: str, text: str, classification: str = "canonical") -> dict:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {
        "path": path,
        "classification": classification,
        "content_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_commit": BASELINE,
        "review_status": "source-reviewed",
        "evidence_tier": "source-contract",
        "maintenance_area": "documentation",
        "scope": "Test maintained guidance, not runtime acceptance.",
        "successors": [],
        "source_references": [path],
        "source_sha256": {path: hashlib.sha256(target.read_bytes()).hexdigest()},
        "validation_commands": [],
    }


def _catalog(*records: dict) -> dict:
    return {
        "schema": catalog_check.SCHEMA,
        "source_baseline": BASELINE,
        "documents": sorted(records, key=lambda row: row["path"]),
        "navigation_sources": [],
        "historical_navigation": [],
    }


def _validate(root: Path, catalog: dict, **kwargs) -> list[str]:
    inventory = {row["path"] for row in catalog["documents"]}
    return catalog_check.validate_catalog(root, catalog, inventory=inventory, **kwargs)


def test_exact_document_bytes_are_required_even_for_historical_records(tmp_path: Path) -> None:
    record = _record(tmp_path, "docs/historical/old.md", "# Old\n", "historical")
    payload = _catalog(record)
    assert _validate(tmp_path, payload) == []
    (tmp_path / record["path"]).write_text("# Changed\n")
    assert any("content changed" in error for error in _validate(tmp_path, payload))


def test_current_inventory_requires_new_documents_and_rejects_deleted_entries(tmp_path: Path) -> None:
    record = _record(tmp_path, "docs/guides/a.md", "# A\n")
    payload = _catalog(record)
    errors = catalog_check.validate_catalog(tmp_path, payload, inventory={"docs/guides/b.md"})
    assert "uncataloged document: docs/guides/b.md" in errors
    assert "catalog entry outside current inventory: docs/guides/a.md" in errors


def test_inventory_includes_prospective_docs_but_not_ignored_outputs(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("docs/generated/\n")
    _record(tmp_path, "docs/guides/prospective.md", "# New\n")
    _record(tmp_path, "docs/generated/ignored.md", "# Generated\n")
    _record(tmp_path, "nested/README.rst", "Title\n=====\n")
    (tmp_path / "unrelated.py").write_text("pass\n")
    assert catalog_check.document_inventory(tmp_path) == {"docs/guides/prospective.md", "nested/README.rst"}


@pytest.mark.parametrize(
    "path", [".", "../outside.md", "/absolute.md", "docs/../alias.md", "docs//alias.md", "docs\\alias.md"]
)
def test_authority_paths_reject_aliases_and_escapes(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError):
        catalog_check.safe_path(tmp_path, path)


def test_authority_paths_reject_internal_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.write_text("target\n")
    (tmp_path / "alias.md").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        catalog_check.safe_path(tmp_path, "alias.md")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("classification", []),
        ("review_status", {}),
        ("evidence_tier", []),
        ("source_references", [{}]),
        ("successors", [{}]),
    ],
)
def test_malformed_metadata_produces_validation_errors(tmp_path: Path, field: str, value: object) -> None:
    record = _record(tmp_path, "docs/guides/a.md", "# A\n")
    record[field] = value
    assert _validate(tmp_path, _catalog(record))


def test_malformed_navigation_exception_produces_validation_error(tmp_path: Path) -> None:
    record = _record(tmp_path, "docs/guides/a.md", "# A\n")
    payload = _catalog(record)
    payload["historical_navigation"] = [{"source": [], "target": {}, "reason": "Malformed"}]
    assert any("historical navigation exception" in error for error in _validate(tmp_path, payload))


def test_successors_must_exist_and_cannot_form_cycles(tmp_path: Path) -> None:
    first = _record(tmp_path, "docs/guides/a.md", "# A\n")
    second = _record(tmp_path, "docs/guides/b.md", "# B\n")
    first["successors"] = [second["path"]]
    second["successors"] = [first["path"]]
    assert any("cycle" in error for error in _validate(tmp_path, _catalog(first, second)))
    second["successors"] = ["docs/guides/renamed.md"]
    assert any("unknown successor" in error for error in _validate(tmp_path, _catalog(first, second)))
    second["successors"] = []
    assert _validate(tmp_path, _catalog(first, second)) == []


def test_review_metadata_cannot_be_inferred_from_classification(tmp_path: Path) -> None:
    record = _record(tmp_path, "docs/guides/a.md", "# A\n")
    record["source_references"] = []
    errors = _validate(tmp_path, _catalog(record))
    assert any("source-reviewed requires" in error for error in errors)
    record["review_status"] = "inherited-classification"
    record["evidence_tier"] = "inventory-only"
    assert _validate(tmp_path, _catalog(record)) == []


def test_unchanged_prose_cannot_retain_authority_when_its_source_changes(tmp_path: Path) -> None:
    record = _record(tmp_path, "docs/guides/a.md", "# A\n")
    source = tmp_path / "implementation.py"
    source.write_text("VERSION = 1\n")
    record["source_references"] = ["implementation.py"]
    record["source_sha256"] = {"implementation.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    payload = _catalog(record)
    assert _validate(tmp_path, payload) == []
    source.write_text("VERSION = 2\n")
    assert any("reviewed source changed" in error for error in _validate(tmp_path, payload))
    catalog_check.refresh_entries(tmp_path, payload, [record["path"]], BASELINE)
    assert _validate(tmp_path, payload) == []


def test_explicit_hash_refresh_never_promotes_review_or_classification(tmp_path: Path) -> None:
    record = _record(tmp_path, "docs/historical/a.md", "# A\n", "historical")
    record["review_status"] = "historical-evidence"
    payload = _catalog(record)
    (tmp_path / record["path"]).write_text("# A\n\nCorrected transcription.\n")
    catalog_check.refresh_entries(tmp_path, payload, [record["path"]], "b" * 40)
    assert record["classification"] == "historical"
    assert record["review_status"] == "historical-evidence"
    assert record["source_commit"] == "b" * 40
    assert _validate(tmp_path, payload) == []
    with pytest.raises(ValueError, match="add/classify"):
        catalog_check.refresh_entries(tmp_path, payload, ["docs/new.md"], BASELINE)


def test_catalog_duplicate_keys_and_unknown_schema_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text('{"schema":"tp.documentation.catalog.v1","schema":"other"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        catalog_check.load_catalog(path)
    path.write_text('{"schema":"other"}')
    with pytest.raises(ValueError, match="expected catalog schema"):
        catalog_check.load_catalog(path)


def test_catalog_self_record_cannot_assert_reviewed_content(tmp_path: Path) -> None:
    record = _record(tmp_path, catalog_check.CATALOG_PATH, "{}\n")
    assert any("self-record" in error for error in _validate(tmp_path, _catalog(record)))
    record.update(
        content_sha256=None,
        review_status="generated-snapshot",
        evidence_tier="generated-metadata",
        source_references=[],
        source_sha256={},
    )
    assert _validate(tmp_path, _catalog(record)) == []


def test_links_cover_inline_reference_shortcut_and_html_but_skip_code(tmp_path: Path) -> None:
    target = _record(tmp_path, "docs/guides/target.md", "# Target\n\n## Valid `Code`\n")
    source = _record(
        tmp_path,
        "docs/guides/source.md",
        "# Source\n[inline](target.md#valid-code)\n"
        "[named][ref]\n[ref]\n[ref]: target.md#valid-code\n"
        '<a href="target.md#valid-code">HTML</a>\n'
        "```md\n[example](absent.md)\n```\n"
        "    [indented](absent.md)\n`[inline code](absent.md)`\n"
        "<!-- [comment](absent.md) -->\n",
    )
    assert _validate(tmp_path, _catalog(source, target)) == []
    links = catalog_check.local_links((tmp_path / source["path"]).read_text())
    assert len(links) == 4


def test_missing_local_targets_and_same_page_anchors_are_blocking(tmp_path: Path) -> None:
    source = _record(tmp_path, "docs/guides/a.md", "# A\n[missing](missing.md)\n[bad](#unknown)\n")
    errors = _validate(tmp_path, _catalog(source))
    assert any("missing local link" in error for error in errors)
    assert any("missing heading #unknown" in error for error in errors)


@pytest.mark.parametrize(
    ("markdown", "destination"),
    [
        (r"[Guide](guide\(draft\).md)", r"guide\(draft\).md"),
        (r"[Guide](guide\].md)", r"guide\].md"),
        (r"[Guide](guide\!.md)", r"guide\!.md"),
        (r"[Guide](guide\\draft.md)", r"guide\\draft.md"),
        ('[Guide](guide.md "Quoted title")', "guide.md"),
        ("[Guide](guide.md 'Quoted title')", "guide.md"),
        ('[Guide](<guide with spaces.md> "Title")', "guide with spaces.md"),
        (r"![Image](image\(draft\).png)", r"image\(draft\).png"),
    ],
)
def test_inline_link_destination_escape_and_title_compatibility(markdown: str, destination: str) -> None:
    assert catalog_check.local_links(markdown) == [(1, destination)]


@pytest.mark.parametrize("escape", [r"\!", "\\", r"\)"], ids=["punctuation", "backslash", "closing-parenthesis"])
def test_unterminated_escaped_destinations_do_not_backtrack_exponentially(escape: str) -> None:
    # Execute both public link validation and navigation labeling in a child so
    # the original exponential pattern fails with a bounded timeout, not a hang.
    text = "[broken](" + escape * 10_000 + "\n[OK](target.md)\n"
    program = (
        "import sys\n"
        "from scripts.governance.check_documentation_catalog import local_links, _visible_navigation_context\n"
        "text = sys.stdin.read()\n"
        "assert local_links(text) == [(2, 'target.md')]\n"
        "assert _visible_navigation_context(text)[2] == '[OK]'\n"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        input=text,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_current_navigation_requires_explicit_dated_evidence_routing(tmp_path: Path) -> None:
    index = _record(tmp_path, "docs/README.md", "# Docs\n[Prior audit](historical/audit.md)\n")
    audit = _record(tmp_path, "docs/historical/audit.md", "# Prior audit\n", "historical")
    payload = _catalog(index, audit)
    payload["navigation_sources"] = [index["path"]]
    assert any("promotes non-maintained" in error for error in _validate(tmp_path, payload))
    payload["historical_navigation"] = [{"source": index["path"], "target": audit["path"], "reason": "Explicit prior audit."}]
    assert _validate(tmp_path, payload) == []
    index["classification"] = "historical"
    assert any("stale historical navigation" in error for error in _validate(tmp_path, payload))


def test_catalog_serialization_is_deterministic(tmp_path: Path) -> None:
    record = _record(tmp_path, "docs/a.md", "# A\n")
    target = tmp_path / "catalog.json"
    catalog_check.write_catalog(target, _catalog(record))
    first = target.read_bytes()
    catalog_check.write_catalog(target, json.loads(first))
    assert target.read_bytes() == first
    assert not list(tmp_path.glob(".documentation-catalog-*"))


@pytest.mark.parametrize(
    ("link", "passes"),
    [
        ("[Production setup](historical/audit.md)", False),
        ("[Prior setup audit](historical/audit.md)", True),
        ('<a href="historical/audit.md">Production setup</a>', False),
        ('<a href="historical/audit.md">Setup</a> — historical evidence', True),
        ("[Setup](historical/audit.md) — 2026-09-12 snapshot", True),
    ],
)
def test_historical_exception_requires_visible_qualifier(tmp_path: Path, link: str, passes: bool) -> None:
    index = _record(tmp_path, "docs/README.md", f"# Docs\n{link}\n")
    audit = _record(tmp_path, "docs/historical/audit.md", "# Prior audit\n", "historical")
    payload = _catalog(index, audit)
    payload["navigation_sources"] = [index["path"]]
    payload["historical_navigation"] = [
        {"source": index["path"], "target": audit["path"], "reason": "Retained historical evidence."}
    ]
    errors = _validate(tmp_path, payload)
    if passes:
        assert errors == []
    else:
        assert any("visible dated/evidence qualifier" in error for error in errors)


def test_invalid_utf8_cannot_silently_skip_maintained_link_validation(tmp_path: Path) -> None:
    record = _record(tmp_path, "docs/guides/a.md", "# Guide\n[Missing](missing.md)\n")
    path = tmp_path / record["path"]
    path.write_bytes(path.read_bytes() + b"\xff")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record.update(content_sha256=digest, source_sha256={record["path"]: digest})
    errors = _validate(tmp_path, _catalog(record))
    assert any("cannot read maintained document for link validation" in error for error in errors)


@pytest.mark.parametrize(
    "example",
    [
        '```html\n<a id="example-only"></a>\n```\n',
        '<!-- <a id="example-only"></a> -->\n',
        '`<a id="example-only"></a>`\n',
        '    <a id="example-only"></a>\n',
    ],
)
def test_html_anchor_examples_do_not_authorize_fragments(tmp_path: Path, example: str) -> None:
    source = _record(tmp_path, "docs/source.md", "# Source\n[Target](target.md#example-only)\n")
    target = _record(tmp_path, "docs/target.md", "# Target\n" + example)
    assert any("missing heading #example-only" in error for error in _validate(tmp_path, _catalog(source, target)))
    target = _record(tmp_path, "docs/target.md", '# Target\n<a id="example-only"></a>\n')
    assert _validate(tmp_path, _catalog(source, target)) == []


@pytest.mark.parametrize("field", ["source_baseline", "source_commit"])
@pytest.mark.parametrize("value", [int("1" * 40), None, [], {}])
def test_source_commit_fields_require_sha_strings(tmp_path: Path, field: str, value: object) -> None:
    record = _record(tmp_path, "docs/a.md", "# A\n")
    payload = _catalog(record)
    if field == "source_baseline":
        payload[field] = value
    else:
        record[field] = value
    assert any(field in error for error in _validate(tmp_path, payload))
