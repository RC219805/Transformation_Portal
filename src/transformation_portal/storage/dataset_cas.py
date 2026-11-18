"""Content-addressed dataset storage.

This module provides dataset management with CAS backing:
- Datasets as collections of CAS objects
- Dataset manifests with logical paths
- Deduplication across datasets and runs
- Efficient materialization

Example:
    >>> dataset_cas = DatasetCAS(cas, Path("/data/datasets"))
    >>>
    >>> # Ingest a directory as a dataset
    >>> manifest = dataset_cas.ingest_directory("training_v1", Path("/raw/images"))
    >>>
    >>> # Later: materialize for use
    >>> dataset_cas.materialize(manifest, Path("/runtime/training"))
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from transformation_portal.storage.cas_store import ArtifactStore, CASError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetFile:
    """A file in a dataset.

    Attributes:
        logical_path: Relative path within dataset
        sha256: CAS hash of file content
        size_bytes: File size in bytes
        metadata: Optional file metadata
    """

    logical_path: str
    sha256: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetManifest:
    """Manifest for a dataset.

    Attributes:
        name: Dataset name
        version: Dataset version
        files: List of files in dataset
        created_at: Creation timestamp
        metadata: Dataset metadata
    """

    name: str
    version: str
    files: List[DatasetFile]
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_size_bytes(self) -> int:
        """Total size of all files."""
        return sum(f.size_bytes for f in self.files)

    @property
    def file_count(self) -> int:
        """Number of files."""
        return len(self.files)

    @property
    def unique_shas(self) -> Set[str]:
        """Set of unique SHA256 hashes."""
        return {f.sha256 for f in self.files}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "files": [asdict(f) for f in self.files],
            "created_at": self.created_at,
            "metadata": self.metadata,
            "summary": {
                "file_count": self.file_count,
                "total_size_bytes": self.total_size_bytes,
                "unique_objects": len(self.unique_shas),
            },
        }

    def to_json(self, *, pretty: bool = True) -> str:
        """Convert to JSON string."""
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: Path) -> None:
        """Save manifest to file."""
        path.write_text(self.to_json())
        logger.info("Saved dataset manifest to %s", path)

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        """Load manifest from file."""
        data = json.loads(path.read_text())

        files = [
            DatasetFile(
                logical_path=f["logical_path"],
                sha256=f["sha256"],
                size_bytes=f["size_bytes"],
                metadata=f.get("metadata", {}),
            )
            for f in data.get("files", [])
        ]
        for file in files:
            _logical_parts(file.logical_path)

        return cls(
            name=data["name"],
            version=data["version"],
            files=files,
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
        )


class DatasetCASError(RuntimeError):
    """Raised for dataset CAS errors."""


def _logical_parts(logical_path: Any) -> tuple[str, ...]:
    """Return the components of a manifest path that stays inside its dataset.

    A logical path names a location *within* a dataset and is later joined
    under a caller-chosen directory. Absolute paths, drive or UNC anchors,
    ``..`` components, NUL bytes and empty paths can only point elsewhere, so
    they are never valid. A whitespace-only file name is unusual but stays
    inside the dataset (``ingest_directory`` records such files verbatim), so
    it is accepted.
    """
    if not isinstance(logical_path, str) or not logical_path or "\x00" in logical_path:
        raise DatasetCASError(f"Invalid dataset logical path: {logical_path!r}")
    candidate = Path(logical_path)
    if candidate.is_absolute() or candidate.anchor or ".." in candidate.parts or not candidate.parts:
        raise DatasetCASError(f"Dataset logical path escapes its dataset: {logical_path!r}")
    return candidate.parts


def _confined_destination(root: Path, logical_path: Any) -> Path:
    """Join ``logical_path`` under the resolved ``root``, refusing any escape.

    The lexical check alone is not enough: a directory symlink already inside
    the target would redirect the write. The parent is therefore resolved and
    must still be inside ``root``. The final component is never followed; the
    CAS replaces it instead.
    """
    destination = root.joinpath(*_logical_parts(logical_path))
    if not destination.parent.resolve().is_relative_to(root):
        raise DatasetCASError(f"Dataset logical path escapes the target directory: {logical_path!r}")
    return destination


class DatasetCAS:
    """Content-addressed dataset storage.

    Manages datasets as collections of CAS objects with manifests.
    Provides deduplication across datasets and efficient materialization.

    Example:
        >>> cas = ArtifactStore(Path("/data/cas"))
        >>> dataset_cas = DatasetCAS(cas, Path("/data/datasets"))
        >>>
        >>> # Ingest directory
        >>> manifest = dataset_cas.ingest_directory(
        ...     name="images_v1",
        ...     src_dir=Path("/raw/training"),
        ...     version="1.0.0",
        ... )
        >>>
        >>> # Check deduplication
        >>> print(f"Files: {manifest.file_count}")
        >>> print(f"Unique objects: {len(manifest.unique_shas)}")
        >>>
        >>> # Materialize for use
        >>> dataset_cas.materialize(manifest, Path("/runtime/training"))
    """

    def __init__(
        self,
        cas: ArtifactStore,
        dataset_root: Path,
    ) -> None:
        """Initialize dataset CAS.

        Args:
            cas: CAS store for objects
            dataset_root: Root directory for dataset manifests
        """
        self.cas = cas
        self.dataset_root = dataset_root
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        logger.info("DatasetCAS initialized: root=%s", dataset_root)

    def _manifest_path(self, name: str, version: str) -> Path:
        """Get path for dataset manifest."""
        # Sanitize name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        safe_version = "".join(c if c.isalnum() or c in "_-." else "_" for c in version)
        return self.dataset_root / f"{safe_name}_{safe_version}.json"

    @staticmethod
    def _manifest_owner(path: Path) -> Optional[tuple[str, str]]:
        """Return the ``(name, version)`` recorded in the manifest at ``path``.

        ``None`` when there is no file or it is not a readable manifest.
        """
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        name, version = data.get("name"), data.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            return None
        return name, version

    def _writable_manifest_path(self, name: str, version: str) -> Path:
        """Return the manifest path for ``(name, version)``, refusing collisions.

        Sanitization maps distinct pairs onto one file name (``train/v1`` and
        ``train_v1``; ``a_b``/``1`` and ``a``/``b_1``). Saving one of them must
        not silently replace the manifest of another dataset.
        """
        path = self._manifest_path(name, version)
        owner = self._manifest_owner(path)
        if owner is not None and owner != (name, version):
            raise DatasetCASError(
                f"Dataset {name!r} v{version!r} collides with existing dataset "
                f"{owner[0]!r} v{owner[1]!r} at {path.name}; choose a different name or version"
            )
        return path

    def ingest_directory(
        self,
        name: str,
        src_dir: Path,
        *,
        version: str = "1.0.0",
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetManifest:
        """Ingest a directory as a dataset.

        Args:
            name: Dataset name
            src_dir: Source directory to ingest
            version: Dataset version
            include_patterns: Glob patterns to include (default: all)
            exclude_patterns: Glob patterns to exclude
            metadata: Additional dataset metadata

        Returns:
            DatasetManifest
        """
        if not src_dir.exists():
            raise DatasetCASError(f"Source directory not found: {src_dir}")
        # Refuse a manifest-path collision before anything is added to the CAS.
        self._writable_manifest_path(name, version)

        files: List[DatasetFile] = []
        total_bytes = 0
        dedup_bytes = 0

        # Collect files
        for path in src_dir.rglob("*"):
            if not path.is_file():
                continue

            rel_path = str(path.relative_to(src_dir))

            # Check include/exclude patterns
            if include_patterns:
                if not any(path.match(p) for p in include_patterns):
                    continue

            if exclude_patterns:
                if any(path.match(p) for p in exclude_patterns):
                    continue

            # Add to CAS
            try:
                obj = self.cas.add_file(path)
                size = path.stat().st_size
                total_bytes += size

                # Check if this was deduplicated
                if obj.path.stat().st_size == size:
                    dedup_bytes += size

                files.append(
                    DatasetFile(
                        logical_path=rel_path,
                        sha256=obj.sha256,
                        size_bytes=size,
                    )
                )

            except CASError as e:
                logger.warning("Failed to add %s to CAS: %s", rel_path, e)

        # Create manifest
        manifest = DatasetManifest(
            name=name,
            version=version,
            files=files,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "source_dir": str(src_dir),
                **(metadata or {}),
            },
        )

        # Save manifest
        manifest_path = self._writable_manifest_path(name, version)
        manifest.save(manifest_path)

        unique_count = len(manifest.unique_shas)
        logger.info(
            "Ingested dataset %s v%s: %d files, %d unique objects, %d bytes",
            name,
            version,
            len(files),
            unique_count,
            total_bytes,
        )

        return manifest

    def ingest_files(
        self,
        name: str,
        files: List[tuple[str, Path]],
        *,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetManifest:
        """Ingest specific files as a dataset.

        Args:
            name: Dataset name
            files: List of (logical_path, source_path) tuples
            version: Dataset version
            metadata: Additional metadata

        Returns:
            DatasetManifest
        """
        dataset_files: List[DatasetFile] = []

        # Refuse an unusable manifest before anything is added to the CAS.
        for logical_path, _src_path in files:
            _logical_parts(logical_path)
        self._writable_manifest_path(name, version)

        for logical_path, src_path in files:
            if not src_path.exists():
                logger.warning("File not found, skipping: %s", src_path)
                continue

            obj = self.cas.add_file(src_path)
            dataset_files.append(
                DatasetFile(
                    logical_path=logical_path,
                    sha256=obj.sha256,
                    size_bytes=src_path.stat().st_size,
                )
            )

        manifest = DatasetManifest(
            name=name,
            version=version,
            files=dataset_files,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

        manifest_path = self._writable_manifest_path(name, version)
        manifest.save(manifest_path)

        return manifest

    def materialize(
        self,
        manifest: DatasetManifest,
        target_dir: Path,
        *,
        use_symlinks: bool = True,
        overwrite: bool = True,
    ) -> Path:
        """Materialize dataset at target location.

        Args:
            manifest: Dataset manifest
            target_dir: Target directory
            use_symlinks: If True, use symlinks instead of copying
            overwrite: If True, overwrite existing files

        Returns:
            Target directory path

        Raises:
            DatasetCASError: If any logical path would be written outside
                ``target_dir``. Nothing is materialized in that case.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        root = target_dir.resolve()
        # Check every entry before writing any: a manifest is untrusted input
        # and one hostile path must not leave the dataset half materialized.
        for file in manifest.files:
            _confined_destination(root, file.logical_path)
        materialized = 0

        for file in manifest.files:
            dst = _confined_destination(root, file.logical_path)
            dst.parent.mkdir(parents=True, exist_ok=True)

            try:
                self.cas.materialize(
                    file.sha256,
                    dst,
                    use_symlink=use_symlinks,
                    overwrite=overwrite,
                )
                materialized += 1
            except CASError as e:
                logger.warning("Failed to materialize %s: %s", file.logical_path, e)

        logger.info(
            "Materialized dataset %s: %d/%d files to %s",
            manifest.name,
            materialized,
            len(manifest.files),
            target_dir,
        )

        return target_dir

    def get_manifest(self, name: str, version: str) -> Optional[DatasetManifest]:
        """Get a dataset manifest.

        Args:
            name: Dataset name
            version: Dataset version

        Returns:
            DatasetManifest or None if not found
        """
        path = self._manifest_path(name, version)
        if not path.exists():
            return None
        manifest = DatasetManifest.load(path)
        if (manifest.name, manifest.version) != (name, version):
            # The file belongs to another dataset whose name sanitizes to the
            # same file name; the requested dataset does not exist.
            return None
        return manifest

    def list_datasets(self) -> List[tuple[str, str]]:
        """List all datasets.

        Returns:
            List of (name, version) tuples
        """
        datasets = []
        for path in self.dataset_root.glob("*.json"):
            try:
                manifest = DatasetManifest.load(path)
                datasets.append((manifest.name, manifest.version))
            except Exception as e:
                logger.warning("Failed to load manifest %s: %s", path, e)
        return datasets

    def delete_dataset(self, name: str, version: str) -> bool:
        """Delete a dataset manifest.

        Note: Does not delete CAS objects (may be shared).

        Args:
            name: Dataset name
            version: Dataset version

        Returns:
            True if deleted
        """
        path = self._manifest_path(name, version)
        if path.exists() and self._manifest_owner(path) in (None, (name, version)):
            path.unlink()
            logger.info("Deleted dataset manifest: %s v%s", name, version)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset storage statistics.

        Returns:
            Dictionary with statistics
        """
        datasets = self.list_datasets()
        total_files = 0
        unique_shas: Set[str] = set()

        for name, version in datasets:
            manifest = self.get_manifest(name, version)
            if manifest:
                total_files += manifest.file_count
                unique_shas.update(manifest.unique_shas)

        return {
            "dataset_count": len(datasets),
            "total_files": total_files,
            "unique_objects": len(unique_shas),
            "dataset_root": str(self.dataset_root),
        }
