"""Property-based tests for Merkle and canonical-JSON invariants.

The existing ``test_ct_merkle.py`` pins concrete RFC 9162 vectors; this
file complements that with Hypothesis-driven *invariants* that must hold
for any input. Failures here surface implementation regressions that
golden vectors might not catch — non-deterministic output, leaf-order
sensitivity, single-leaf identity violations, inclusion-proof drift.

Three surfaces are exercised:

* ``tp.crypto.merkle.merkle_root_sha256`` — the simpler duplicate-last
  construction used by deterministic tooling layers.
* ``tp.crypto.ct_merkle`` (root + inclusion proof + verification) — the
  RFC 9162 transparency-grade construction used by evidence chains.
* ``transformation_portal.ingest.canonical_json.canonicalize_json`` —
  the wire format that both Merkle constructions feed from. The
  canonical-JSON invariant (round-trip + key-order independence) is what
  makes Merkle-rooted evidence reproducible across runs.
"""

from __future__ import annotations

import hashlib
import json

import pytest

# Hypothesis is part of the dev requirements (pyproject.toml + requirements/dev.in)
# but skip cleanly if it is somehow absent in a stripped-down env.
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tp.crypto.ct_merkle import (  # noqa: E402
    ct_inclusion_proof,
    ct_leaf_hash,
    ct_merkle_root,
    verify_ct_inclusion_proof,
)
from tp.crypto.merkle import merkle_root_sha256  # noqa: E402
from tp.merkle import merkle_root_sha256 as merkle_root_sha256_reexport  # noqa: E402
from transformation_portal.ingest.canonical_json import canonicalize_json  # noqa: E402

pytestmark = [pytest.mark.unit]


# ``settings`` profile keeps the suite snappy on PR feedback paths while
# still giving Hypothesis enough examples to find counterexamples.
_SETTINGS = settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# A leaf in either Merkle construction is an arbitrary 32-byte digest.
_LEAF_BYTES = st.binary(min_size=32, max_size=32)
_LEAF_LIST = st.lists(_LEAF_BYTES, min_size=1, max_size=64)


class TestSimpleMerkleRoot:
    """Properties of the duplicate-last ``merkle_root_sha256`` construction."""

    def test_empty_root_is_sha256_of_empty_string(self):
        # The contract calls out that an empty leaf list returns
        # SHA256(""). Pin it so a refactor can't quietly change this.
        assert merkle_root_sha256([]) == hashlib.sha256(b"").hexdigest()

    @given(_LEAF_BYTES)
    @_SETTINGS
    def test_single_leaf_root_equals_leaf_hex(self, leaf):
        # With a single leaf, the root must be that leaf's hex value
        # (no further hashing happens — the contract stops at one node).
        assert merkle_root_sha256([leaf]) == leaf.hex()

    @given(_LEAF_LIST)
    @_SETTINGS
    def test_root_is_deterministic(self, leaves):
        # Same input → same output, always. This is the single most
        # important Merkle invariant; a regression here breaks every
        # downstream attestation that pins a root.
        assert merkle_root_sha256(leaves) == merkle_root_sha256(leaves)

    @given(_LEAF_LIST)
    @_SETTINGS
    def test_root_is_64_char_hex(self, leaves):
        root = merkle_root_sha256(leaves)
        assert len(root) == 64
        assert all(c in "0123456789abcdef" for c in root)

    @given(_LEAF_LIST)
    @_SETTINGS
    def test_leaf_order_changes_root(self, leaves):
        # Reversing a non-singleton/non-palindromic leaf list must
        # change the root — Merkle trees are order-sensitive by design.
        # ``assume`` filters out the boring single-leaf and palindromic
        # cases so Hypothesis spends its budget on the interesting space.
        assume(len(leaves) >= 2)
        assume(leaves != list(reversed(leaves)))
        assert merkle_root_sha256(leaves) != merkle_root_sha256(list(reversed(leaves)))

    @given(_LEAF_LIST)
    @_SETTINGS
    def test_re_export_matches_canonical_module(self, leaves):
        # ``tp.merkle`` re-exports ``merkle_root_sha256`` for backward
        # compatibility. Ensure the re-export is the same callable, not
        # a stale copy that could drift.
        assert merkle_root_sha256_reexport(leaves) == merkle_root_sha256(leaves)


class TestCTMerkleRoot:
    """Properties of the RFC 9162 ``ct_merkle_root`` construction."""

    def test_empty_root_is_sha256_of_empty_string(self):
        # RFC 9162 Section 2.1: MTH({}) = SHA-256().
        assert ct_merkle_root([]) == hashlib.sha256(b"").digest()

    @given(_LEAF_BYTES)
    @_SETTINGS
    def test_single_leaf_root_is_leaf(self, leaf):
        # MTH({d(0)}) = leaf_hash; in our API the leaf is already hashed.
        assert ct_merkle_root([leaf]) == leaf

    @given(_LEAF_LIST)
    @_SETTINGS
    def test_root_is_deterministic(self, leaves):
        assert ct_merkle_root(leaves) == ct_merkle_root(leaves)

    @given(_LEAF_LIST)
    @_SETTINGS
    def test_root_is_32_bytes(self, leaves):
        root = ct_merkle_root(leaves)
        assert isinstance(root, bytes)
        assert len(root) == 32

    @given(_LEAF_LIST)
    @_SETTINGS
    def test_leaf_order_changes_root(self, leaves):
        assume(len(leaves) >= 2)
        assume(leaves != list(reversed(leaves)))
        assert ct_merkle_root(leaves) != ct_merkle_root(list(reversed(leaves)))

    @given(_LEAF_LIST, st.data())
    @_SETTINGS
    def test_inclusion_proof_round_trip(self, leaves, data):
        # For every (leaves, leaf_index), building an inclusion proof and
        # then verifying it must succeed — and verification must be
        # constant-time, side-effect free, and return True. This is the
        # core trust contract for evidence/attestation Merkle chains.
        leaf_index = data.draw(st.integers(min_value=0, max_value=len(leaves) - 1))
        root = ct_merkle_root(leaves)
        proof = ct_inclusion_proof(leaves, leaf_index)
        assert verify_ct_inclusion_proof(
            leaf_hash=leaves[leaf_index],
            leaf_index=leaf_index,
            tree_size=len(leaves),
            proof=proof,
            expected_root=root,
        ) is True

    @given(_LEAF_LIST, st.data())
    @_SETTINGS
    def test_corrupted_root_fails_verification(self, leaves, data):
        # Flipping any bit of the expected root must make the proof
        # invalid — fail-closed is the security contract.
        leaf_index = data.draw(st.integers(min_value=0, max_value=len(leaves) - 1))
        root = bytearray(ct_merkle_root(leaves))
        root[0] ^= 0x01
        proof = ct_inclusion_proof(leaves, leaf_index)
        assert verify_ct_inclusion_proof(
            leaf_hash=leaves[leaf_index],
            leaf_index=leaf_index,
            tree_size=len(leaves),
            proof=proof,
            expected_root=bytes(root),
        ) is False

    @given(_LEAF_LIST, st.data())
    @_SETTINGS
    def test_wrong_leaf_index_fails_verification(self, leaves, data):
        # Claiming a different leaf index than the one the proof was
        # built for must not verify.
        assume(len(leaves) >= 2)
        true_index = data.draw(st.integers(min_value=0, max_value=len(leaves) - 1))
        wrong_index = data.draw(
            st.integers(min_value=0, max_value=len(leaves) - 1).filter(lambda i: i != true_index)
        )
        root = ct_merkle_root(leaves)
        proof = ct_inclusion_proof(leaves, true_index)
        assert verify_ct_inclusion_proof(
            leaf_hash=leaves[true_index],
            leaf_index=wrong_index,
            tree_size=len(leaves),
            proof=proof,
            expected_root=root,
        ) is False


class TestCTLeafAndNodeHashing:
    """Domain-separation invariants for CT leaf/node hashing."""

    @given(st.binary(max_size=512))
    @_SETTINGS
    def test_leaf_hash_is_domain_separated_from_raw(self, payload):
        # ``ct_leaf_hash(x)`` MUST NOT equal SHA256(x). Without the 0x00
        # prefix, second-preimage attacks could swap leaves and interior
        # nodes; that's the entire reason RFC 9162 specifies the prefix.
        assert ct_leaf_hash(payload) != hashlib.sha256(payload).digest()

    @given(st.binary(min_size=32, max_size=32), st.binary(min_size=32, max_size=32))
    @_SETTINGS
    def test_leaf_hash_and_node_hash_disjoint(self, left, right):
        # A 32-byte leaf payload and the same bytes used as a (left+right)
        # node concatenation must hash to different values.
        # ``ct_node_hash(l, r) = SHA256(0x01 || l || r)`` vs
        # ``ct_leaf_hash(l + r) = SHA256(0x00 || l || r)``.
        from tp.crypto.ct_merkle import ct_node_hash

        assert ct_leaf_hash(left + right) != ct_node_hash(left, right)


# Hypothesis JSON-value strategy: keep it inside the union of types our
# canonicalizer commits to handling (no NaN/Infinity, no bytes — bytes
# are decoded by ``to_jsonable`` but the round-trip below compares the
# canonical bytes, which means we need types ``json.loads`` can rebuild).
_JSON_PRIMITIVES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=32),
)
_JSON_VALUES = st.recursive(
    _JSON_PRIMITIVES,
    lambda children: st.one_of(
        st.lists(children, max_size=8),
        st.dictionaries(st.text(min_size=1, max_size=16), children, max_size=8),
    ),
    max_leaves=20,
)


class TestCanonicalJsonProperties:
    """Canonical-JSON invariants: round-trip, key-order independence, determinism."""

    @given(_JSON_VALUES)
    @_SETTINGS
    def test_canonicalize_round_trips_through_json(self, payload):
        # Canonical-JSON output must be round-trip-stable: encode →
        # decode → encode produces the same bytes. This is what makes
        # Merkle rooting reproducible across runs.
        encoded = canonicalize_json(payload)
        decoded = json.loads(encoded)
        re_encoded = canonicalize_json(decoded)
        assert encoded == re_encoded

    @given(st.dictionaries(st.text(min_size=1, max_size=16), _JSON_PRIMITIVES, min_size=2, max_size=8))
    @_SETTINGS
    def test_dict_key_order_does_not_change_output(self, payload):
        # The canonical form sorts keys, so two dicts with identical
        # contents in different insertion order must canonicalize to
        # the same bytes. This is the key invariant that lets us hash
        # JSON evidence without normalizing it manually first.
        reordered = {k: payload[k] for k in reversed(list(payload.keys()))}
        assert canonicalize_json(payload) == canonicalize_json(reordered)

    @given(_JSON_VALUES)
    @_SETTINGS
    def test_canonicalize_is_deterministic(self, payload):
        # Same input → same bytes, every call. (Trivially true unless
        # someone introduces a non-deterministic key ordering — pin it.)
        assert canonicalize_json(payload) == canonicalize_json(payload)

    @given(_JSON_VALUES)
    @_SETTINGS
    def test_canonicalize_uses_no_whitespace(self, payload):
        # The canonical profile uses ``separators=(",", ":")`` — no
        # incidental whitespace. Pinning this prevents a careless
        # ``indent=2`` from sneaking in and silently changing every hash.
        encoded = canonicalize_json(payload)
        # The encoded bytes must not contain ``", "`` or ``": "``
        # separator pairs (with the space) anywhere.
        assert b", " not in encoded
        assert b": " not in encoded

    def test_canonicalize_rejects_nan_and_infinity(self):
        # Both NaN and Infinity break JSON-strict consumers and would
        # produce non-deterministic Merkle roots if accepted.
        with pytest.raises(ValueError):
            canonicalize_json({"x": float("nan")})
        with pytest.raises(ValueError):
            canonicalize_json({"x": float("inf")})
        with pytest.raises(ValueError):
            canonicalize_json({"x": float("-inf")})
