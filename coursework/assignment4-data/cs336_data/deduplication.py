"""Deterministic document deduplication helpers for the assignment 4 demo.

The assignment introduces MinHash/LSH for document-level near-duplicate removal.
This module keeps that implementation small enough for a local example while using
stable hashes so that the result does not depend on Python's per-process hash seed.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class MinHashStats:
    """Statistics produced while finding near-duplicate documents."""

    candidate_pairs: int
    duplicate_documents: int


def normalize_for_shingles(text: str) -> str:
    """Normalize text before creating character shingles.

    Case, punctuation, and whitespace differences are ignored.  The normalized
    text is used only for similarity decisions; the original filtered text is
    retained in the output.
    """

    normalized = unicodedata.normalize("NFKC", text).lower()
    chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if char.isspace() or category.startswith(("P", "S")):
            chars.append(" ")
        else:
            chars.append(char)
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _character_shingles(text: str, size: int) -> set[str]:
    if not text:
        return set()
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _stable_hash(value: str, seed: int) -> int:
    payload = f"{seed}\0{value}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _signature(shingles: set[str], num_hashes: int) -> tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(num_hashes))
    return tuple(min(_stable_hash(shingle, seed) for shingle in shingles) for seed in range(num_hashes))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def minhash_deduplicate(
    texts: list[str],
    *,
    num_hashes: int = 64,
    num_bands: int = 16,
    shingle_size: int = 5,
    jaccard_threshold: float = 0.85,
) -> tuple[list[int], MinHashStats]:
    """Return indices to keep after MinHash/LSH near-duplicate removal.

    LSH only generates candidate pairs.  A candidate is removed only after its
    true Jaccard similarity reaches ``jaccard_threshold``.  The first document
    in each connected duplicate cluster is retained.
    """

    if num_hashes <= 0 or num_bands <= 0 or num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be positive and divisible by num_bands")
    if shingle_size <= 0:
        raise ValueError("shingle_size must be positive")
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be between 0 and 1")
    if not texts:
        return [], MinHashStats(candidate_pairs=0, duplicate_documents=0)

    shingle_sets = [_character_shingles(normalize_for_shingles(text), shingle_size) for text in texts]
    signatures = [_signature(shingles, num_hashes) for shingles in shingle_sets]
    rows_per_band = num_hashes // num_bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    candidate_pairs: set[tuple[int, int]] = set()

    for document_index, signature in enumerate(signatures):
        for band_index in range(num_bands):
            start = band_index * rows_per_band
            key = (band_index, signature[start : start + rows_per_band])
            for other_index in buckets[key]:
                candidate_pairs.add((other_index, document_index))
            buckets[key].append(document_index)

    union_find = _UnionFind(len(texts))
    # Sort the set before unioning so cluster representatives do not depend on
    # Python's set iteration order.
    for left, right in sorted(candidate_pairs):
        if _jaccard(shingle_sets[left], shingle_sets[right]) >= jaccard_threshold:
            union_find.union(left, right)

    kept_indices: list[int] = []
    seen_roots: set[int] = set()
    for index in range(len(texts)):
        root = union_find.find(index)
        if root not in seen_roots:
            seen_roots.add(root)
            kept_indices.append(index)

    return kept_indices, MinHashStats(
        candidate_pairs=len(candidate_pairs),
        duplicate_documents=len(texts) - len(kept_indices),
    )
