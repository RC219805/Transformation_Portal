"""Cryptographic helpers for deterministic contract surfaces.

This package provides Merkle tree implementations for integrity verification:

- merkle_root_sha256: Simple duplicate-last pairing (legacy/provenance use)
- ct_* functions: RFC 9162 Certificate Transparency-style trees with
  verifiable inclusion and consistency proofs (artifact trees, evidence bundles)

Type aliases for API clarity:
- Sha256Digest: 32-byte SHA-256 digest
- MerkleRoot: 32-byte Merkle tree root
- AuditPath: List of sibling digests for inclusion proofs
- ConsistencyPath: List of sibling digests for consistency proofs
"""

from .ct_merkle import (  # Type aliases; Validation; CT-style Merkle functions
    AuditPath,
    ConsistencyPath,
    MerkleRoot,
    Sha256Digest,
    ct_consistency_proof,
    ct_consistency_proof_sha256,
    ct_inclusion_proof,
    ct_inclusion_proof_sha256,
    ct_leaf_hash,
    ct_merkle_root,
    ct_merkle_root_sha256,
    ct_node_hash,
    validate_sha256_digest,
    verify_ct_consistency_proof,
    verify_ct_inclusion_proof,
)
from .merkle import merkle_root_sha256

__all__ = [
    # Type aliases
    "Sha256Digest",
    "MerkleRoot",
    "AuditPath",
    "ConsistencyPath",
    # Validation
    "validate_sha256_digest",
    # Legacy/provenance Merkle
    "merkle_root_sha256",
    # CT-style Merkle (RFC 9162)
    "ct_leaf_hash",
    "ct_node_hash",
    "ct_merkle_root",
    "ct_merkle_root_sha256",
    "ct_inclusion_proof",
    "ct_inclusion_proof_sha256",
    "verify_ct_inclusion_proof",
    "ct_consistency_proof",
    "ct_consistency_proof_sha256",
    "verify_ct_consistency_proof",
]
