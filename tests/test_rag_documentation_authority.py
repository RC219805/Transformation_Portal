"""Documentation authority must follow reviewed bytes rather than relevance scores."""

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
AGENTS = Path(__file__).resolve().parents[1] / ".github" / "agents"
sys.path.insert(0, str(AGENTS))

from rag_system.authority import CATALOG_PATH, DocumentationCatalog  # noqa: E402
from rag_system.citation import CitationGenerator  # noqa: E402
from rag_system.indexer import RepositoryIndexer  # noqa: E402
from rag_system.reranker import ResultReranker  # noqa: E402
from rag_system.retriever import HybridRetriever  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_catalog(root, entries):
    path = root / CATALOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "tp.documentation.catalog.v1", "source_baseline": "a" * 40, "documents": entries}))
    return path


def entry(root, path, classification="canonical", successors=None):
    return {
        "path": path,
        "classification": classification,
        "content_sha256": digest(root / path),
        "source_commit": "a" * 40,
        "review_status": "source-reviewed",
        "evidence_tier": "source-contract",
        "maintenance_area": "documentation",
        "scope": "Synthetic local contract fixture; no runtime acceptance",
        "successors": successors or [],
        "source_references": ["Makefile"],
        "source_sha256": {"Makefile": digest(root / "Makefile")},
    }


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "Makefile").write_text("test: # governed source outside retrieval inventory\n")
    (tmp_path / "docs/current.md").write_text("# Guide\nDeployment identity configuration. " + "context " * 100)
    (tmp_path / "docs/old.md").write_text("# Deployment identity\nDeployment identity deployment identity.")
    entries = [entry(tmp_path, "docs/current.md"), entry(tmp_path, "docs/old.md", "historical", ["docs/current.md"])]
    write_catalog(tmp_path, entries)
    return tmp_path, entries


def docs(indexer):
    return {chunk.file_path: chunk for chunk in indexer.index_repository()}


def test_operator_prefers_reviewed_guidance_and_explicit_modes_keep_history(corpus):
    root, _ = corpus
    chunks = RepositoryIndexer(root, use_cache=False).index_repository()
    retriever = HybridRetriever(enable_vector_search=False)
    retriever.index(chunks)
    operator = retriever.retrieve("deployment identity", top_k=2)
    assert [result.file_path for result in operator] == ["docs/current.md", "docs/old.md"]
    assert retriever.retrieve("deployment identity", top_k=1)[0].file_path == "docs/current.md"
    assert retriever.retrieve("deployment identity", top_k=1, retrieval_mode="all")[0].file_path == "docs/old.md"
    history = retriever.retrieve("deployment identity", top_k=5, retrieval_mode="historical")
    assert [result.file_path for result in history] == ["docs/old.md"]
    reranked = ResultReranker().rerank(operator, "deployment identity")
    assert reranked[0].file_path == "docs/current.md"
    generator = CitationGenerator()
    rendered = generator.format_citations(generator.generate_citations(history))
    assert "historical (verified)" in rendered
    assert "successors: docs/current.md" in rendered
    assert "not correctness or authority" in rendered
    assert all(CATALOG_PATH != chunk.file_path for chunk in chunks)


@pytest.mark.parametrize(
    "missing", ["classification", "content_sha256", "review_status", "source_references", "source_sha256"]
)
def test_incomplete_catalog_entries_never_authorize(corpus, missing):
    root, entries = corpus
    del entries[0][missing]
    write_catalog(root, entries)
    authority = docs(RepositoryIndexer(root, use_cache=False))["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"


@pytest.mark.parametrize("malformed", [{}, [], None, 1])
def test_malformed_classification_fails_closed(corpus, malformed):
    root, entries = corpus
    entries[0]["classification"] = malformed
    write_catalog(root, entries)
    assert (
        docs(RepositoryIndexer(root, use_cache=False))["docs/current.md"].metadata["documentation"]["authority"]
        == "unverified"
    )


@pytest.mark.parametrize("state", ["missing", "invalid", "wrong-schema"])
def test_unavailable_catalog_keeps_results_without_authority(corpus, state):
    root, _ = corpus
    path = root / CATALOG_PATH
    if state == "missing":
        path.unlink()
    elif state == "invalid":
        path.write_text("not JSON")
    else:
        path.write_text(json.dumps({"schema": "unknown", "documents": []}))
    chunks = RepositoryIndexer(root, use_cache=False).index_repository()
    assert len(chunks) == 2
    assert all(chunk.metadata["documentation"]["authority"] == "unverified" for chunk in chunks)
    retriever = HybridRetriever(enable_vector_search=False)
    retriever.index(chunks)
    assert len(retriever.retrieve("deployment identity", retrieval_mode="all")) == 2


def test_unreviewed_current_classification_is_not_maintained(corpus):
    root, entries = corpus
    entries[0]["review_status"] = "inherited-classification"
    write_catalog(root, entries)
    authority = docs(RepositoryIndexer(root, use_cache=False))["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "unreviewed"


def test_stale_document_hash_does_not_authorize(corpus):
    root, _ = corpus
    (root / "docs/current.md").write_text("# Deployment identity changed without review")
    authority = docs(RepositoryIndexer(root))["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "stale-content"


def test_source_outside_retrieval_inventory_invalidates_cached_authority(corpus):
    root, _ = corpus
    indexer = RepositoryIndexer(root)
    assert docs(indexer)["docs/current.md"].metadata["documentation"]["authority"] == "maintained"
    (root / "Makefile").write_text("test: # changed contract without review\n")
    authority = docs(indexer)["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "stale-source"


def test_catalog_only_review_change_invalidates_cache(corpus):
    root, entries = corpus
    indexer = RepositoryIndexer(root)
    first = docs(indexer)
    assert first["docs/current.md"].metadata["documentation"]["authority"] == "maintained"
    entries[0]["review_status"] = "inherited-classification"
    write_catalog(root, entries)
    second = docs(indexer)
    assert second["docs/current.md"].content == first["docs/current.md"].content
    assert second["docs/current.md"].metadata["documentation"]["authority"] == "unverified"


def test_renamed_successor_requires_explicit_catalog_update(corpus):
    root, entries = corpus
    indexer = RepositoryIndexer(root)
    docs(indexer)
    (root / "docs/current.md").rename(root / "docs/successor.md")
    authority = docs(indexer)["docs/old.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "unresolved-successor"
    entries[0]["path"] = "docs/successor.md"
    entries[1]["successors"] = ["docs/successor.md"]
    write_catalog(root, entries)
    authority = docs(indexer)["docs/old.md"].metadata["documentation"]
    assert authority["authority"] == "historical"
    assert authority["successors"] == ["docs/successor.md"]


def test_catalog_transient_a_b_a_does_not_publish_b_as_a(corpus, monkeypatch):
    root, entries = corpus
    indexer = RepositoryIndexer(root)
    path = root / CATALOG_PATH
    original_bytes = path.read_bytes()
    read = DocumentationCatalog.read
    calls = 0

    def changing_read(repo_root):
        nonlocal calls
        calls += 1
        if calls == 1:
            entries[0]["review_status"] = "inherited-classification"
            write_catalog(root, entries)
            snapshot = read(repo_root)
            path.write_bytes(original_bytes)
            return snapshot
        return read(repo_root)

    monkeypatch.setattr(DocumentationCatalog, "read", changing_read)
    authority = docs(indexer)["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert not indexer.cache_file.exists()
    assert path.read_bytes() == original_bytes


def test_reindex_clears_retrieval_authority_query_cache(corpus):
    root, entries = corpus
    indexer = RepositoryIndexer(root)
    retriever = HybridRetriever(enable_vector_search=False)
    retriever.index(indexer.index_repository())
    assert retriever.retrieve("deployment identity", top_k=1)[0].file_path == "docs/current.md"
    entries[0]["review_status"] = "inherited-classification"
    write_catalog(root, entries)
    retriever.index(indexer.index_repository())
    assert retriever.retrieve("deployment identity", top_k=1)[0].file_path == "docs/old.md"


def test_template_readme_python_integration_is_executable(corpus, monkeypatch):
    root, entries = corpus
    (root / "docs/current.md").write_text("# Atmospheric effects depth processing")
    entries[0]["content_sha256"] = digest(root / "docs/current.md")
    write_catalog(root, entries)
    monkeypatch.chdir(root)
    text = (AGENTS / "rag_system/templates/README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", text, flags=re.S)
    code = next(block for block in blocks if "from rag_system.indexer import RepositoryIndexer" in block)
    namespace = {}
    exec(compile(code, "templates/README.md", "exec"), namespace)
    assert namespace["template_with_examples"]
    assert namespace["citations"]
    assert "Citations" in namespace["retrieved_context"]


def test_testing_template_uses_registered_markers():
    import tomllib

    repo = AGENTS.parents[1]
    configuration = tomllib.loads((repo / "pyproject.toml").read_text())
    registered = {value.split(":", 1)[0] for value in configuration["tool"]["pytest"]["ini_options"]["markers"]}
    text = (AGENTS / "rag_system/templates/testing.md").read_text()
    markers = set(re.findall(r"@pytest\.mark\.([a-z_]+)", text)) - {"parametrize", "skipif", "skip", "xfail"}
    assert markers <= registered


def test_source_provenance_transient_a_b_a_does_not_publish_stale_authority(corpus, monkeypatch):
    root, _ = corpus
    indexer = RepositoryIndexer(root)
    path = root / "Makefile"
    original_bytes = path.read_bytes()
    read = DocumentationCatalog.read
    calls = 0

    def changing_read(repo_root):
        nonlocal calls
        calls += 1
        if calls == 1:
            path.write_text("transient changed contract")
            snapshot = read(repo_root)
            path.write_bytes(original_bytes)
            return snapshot
        return read(repo_root)

    monkeypatch.setattr(DocumentationCatalog, "read", changing_read)
    authority = docs(indexer)["docs/current.md"].metadata["documentation"]
    assert authority["verification"] == "stale-source"
    assert not indexer.cache_file.exists()
    assert path.read_bytes() == original_bytes


def test_missing_external_source_does_not_reuse_maintained_cache(corpus):
    root, _ = corpus
    indexer = RepositoryIndexer(root)
    docs(indexer)
    (root / "Makefile").unlink()
    authority = docs(indexer)["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "stale-source"


def test_historical_cli_mode_is_available(corpus):
    import os
    import subprocess

    root, _ = corpus
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_system.cli",
            "search",
            "deployment identity",
            "--repo-root",
            str(root),
            "--mode",
            "historical",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(AGENTS), "RAG_RETRIEVER_ENABLE_VECTOR_SEARCH": "false"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "[1] docs/old.md:" in completed.stdout
    assert "[1] docs/current.md:" not in completed.stdout


def test_source_review_without_source_contract_evidence_is_unverified(corpus):
    root, entries = corpus
    entries[0]["evidence_tier"] = "inventory-only"
    write_catalog(root, entries)
    authority = docs(RepositoryIndexer(root, use_cache=False))["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "invalid-source-provenance"


@pytest.mark.parametrize("field", ["classification", "review_status", "source_sha256"])
def test_duplicate_json_keys_never_authorize(corpus, field):
    root, _ = corpus
    path = root / CATALOG_PATH
    original = path.read_text()
    path.write_text(original.replace(f'"{field}":', f'"{field}": null, "{field}":', 1))
    authority = docs(RepositoryIndexer(root, use_cache=False))["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "invalid-catalog"


def test_external_catalog_symlink_never_authorizes(corpus, tmp_path):
    root, _ = corpus
    path = root / CATALOG_PATH
    external = tmp_path.parent / f"{tmp_path.name}-outside-catalog.json"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    authority = docs(RepositoryIndexer(root, use_cache=False))["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "unsafe-catalog-path"


def test_same_byte_document_symlink_invalidates_cached_authority(corpus, tmp_path):
    root, _ = corpus
    indexer = RepositoryIndexer(root)
    assert docs(indexer)["docs/current.md"].metadata["documentation"]["authority"] == "maintained"
    path = root / "docs/current.md"
    external = tmp_path.parent / f"{tmp_path.name}-outside-document.txt"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    authority = docs(indexer)["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "unsafe-document-path"


def test_cached_authority_is_recomputed_from_reviewed_catalog(corpus):
    root, entries = corpus
    entries[0]["review_status"] = "inherited-classification"
    write_catalog(root, entries)
    indexer = RepositoryIndexer(root)
    docs(indexer)
    payload = json.loads(indexer.cache_file.read_text())
    for chunk in payload["chunks"]:
        chunk["metadata"]["documentation"].update(authority="maintained", verification="verified")
    indexer.cache_file.write_text(json.dumps(payload))
    authority = docs(indexer)["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "unreviewed"


def test_modified_cached_text_cannot_inherit_source_authority(corpus):
    root, _ = corpus
    indexer = RepositoryIndexer(root)
    expected = docs(indexer)["docs/current.md"].content
    payload = json.loads(indexer.cache_file.read_text())
    for chunk in payload["chunks"]:
        if chunk["file_path"] == "docs/current.md":
            chunk["content"] = "Unreviewed instructions substituted into an otherwise current cache."
    indexer.cache_file.write_text(json.dumps(payload))
    assert docs(indexer)["docs/current.md"].content == expected


@pytest.mark.parametrize("self_cycle", [True, False])
def test_cyclic_successor_closure_never_authorizes(corpus, self_cycle):
    root, entries = corpus
    entries[0]["successors"] = ["docs/current.md" if self_cycle else "docs/old.md"]
    write_catalog(root, entries)
    authority = docs(RepositoryIndexer(root, use_cache=False))["docs/current.md"].metadata["documentation"]
    assert authority["authority"] == "unverified"
    assert authority["verification"] == "invalid-catalog"


def test_cached_code_cannot_gain_documentation_authority(corpus):
    root, _ = corpus
    (root / "src").mkdir()
    (root / "src/example.py").write_text(
        'def example():\n    """A source function without documentation review authority."""\n    return True\n'
    )
    indexer = RepositoryIndexer(root)
    docs(indexer)
    payload = json.loads(indexer.cache_file.read_text())
    for chunk in payload["chunks"]:
        if chunk["file_path"] == "src/example.py":
            chunk["metadata"]["documentation"] = {"authority": "maintained", "verification": "verified"}
    indexer.cache_file.write_text(json.dumps(payload))
    assert "documentation" not in docs(indexer)["src/example.py"].metadata


@pytest.mark.parametrize("cache_state", ["disabled", "cold", "warm"])
@pytest.mark.parametrize("changed", ["source", "catalog", "document"])
@pytest.mark.parametrize("current_path", ["docs/current.md", "AGENTS.md"])
def test_returned_authority_is_withheld_when_final_snapshot_changes(corpus, monkeypatch, cache_state, changed, current_path):
    """Known mid-index drift cannot promote an old snapshot in operator queries."""
    root, entries = corpus
    if current_path == "AGENTS.md":
        (root / "docs/current.md").rename(root / current_path)
        entries[0]["path"] = current_path
        entries[1]["successors"] = [current_path]
        write_catalog(root, entries)
    indexer = RepositoryIndexer(root, use_cache=cache_state != "disabled")
    if cache_state == "warm":
        indexer.index_repository()
        assert indexer.cache_file.exists()
    metadata = DocumentationCatalog.metadata
    mutated = False

    def edit_after_metadata(snapshot, path, source_hash):
        nonlocal mutated
        result = metadata(snapshot, path, source_hash)
        if path == current_path and not mutated:
            assert result["authority"] == "maintained"
            if changed == "source":
                (root / "Makefile").write_text("test: # unreviewed implementation change\n")
            elif changed == "catalog":
                entries[0]["review_status"] = "inherited-classification"
                write_catalog(root, entries)
            else:
                (root / current_path).write_text("# Changed\nUnreviewed guidance.\n")
            mutated = True
        return result

    monkeypatch.setattr(DocumentationCatalog, "metadata", edit_after_metadata)
    chunks = indexer.index_repository()
    assert mutated
    current = next(chunk for chunk in chunks if chunk.file_path == current_path)
    assert current.metadata["documentation"]["authority"] == "unverified"
    assert current.metadata["documentation"]["verification"] == "changed-during-indexing"
    if cache_state != "warm":
        assert not indexer.cache_file.exists()
    retriever = HybridRetriever(enable_vector_search=False)
    retriever.index(chunks)
    operator = retriever.retrieve("deployment identity", top_k=2)
    assert operator[0].file_path == "docs/old.md"
    assert all(result.metadata["documentation"]["authority"] != "maintained" for result in operator)
    assert ResultReranker().rerank(operator, "deployment identity")[0].file_path == "docs/old.md"
