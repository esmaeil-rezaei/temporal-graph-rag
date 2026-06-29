from __future__ import annotations

import hashlib

from datasketch import MinHash, MinHashLSH

from src.config.settings import get_config
from src.ingestion.parser import ParsedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Deduplicator:
    """
    In-memory exact + fuzzy deduplication for ingestion runs.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._dedup_cfg = cfg.ingestion["deduplication"]

        self._num_perm: int = self._dedup_cfg["minhash_num_perm"]
        self._threshold: float = self._dedup_cfg["jaccard_threshold"]
        self._keep_strategy: str = self._dedup_cfg["keep_strategy"]

        self._exact_hashes: set[str] = set()

        self._lsh = MinHashLSH(
            threshold=self._threshold,
            num_perm=self._num_perm,
        )
        self._lsh_registry: dict[str, tuple[ParsedChunk, MinHash]] = {}

    def filter(self, chunks: list[ParsedChunk]) -> list[ParsedChunk]:
        """
        Deduplicate a batch of chunks using exact and MinHash matching.
        """
        if not self._dedup_cfg["enabled"]:
            return chunks

        unique: list[ParsedChunk] = []
        duplicates_exact = 0
        duplicates_fuzzy = 0

        for chunk in chunks:

            norm_text = self._normalize(chunk.text)
            exact_hash = self._compute_exact_hash(norm_text)
            if exact_hash in self._exact_hashes:
                duplicates_exact += 1
                logger.debug(f"Exact duplicate skipped: {chunk.chunk_id}")
                continue

            minhash = self._compute_minhash(norm_text)
            similar_keys: list[str] = self._lsh.query(minhash)

            if similar_keys:
                winner = self._resolve_conflict(chunk, similar_keys)
                if winner is not chunk:
                    duplicates_fuzzy += 1
                    logger.debug(
                        f"Near-duplicate skipped (jaccard ≥ {self._threshold}): "
                        f"{chunk.chunk_id}"
                    )
                    continue

            self._exact_hashes.add(exact_hash)
            lsh_key = chunk.chunk_id or exact_hash
            try:
                self._lsh.insert(lsh_key, minhash)
            except ValueError:
                pass
            self._lsh_registry[lsh_key] = (chunk, minhash)
            unique.append(chunk)

        logger.info(
            f"Deduplication complete: {len(unique)} unique, "
            f"{duplicates_exact} exact dups, {duplicates_fuzzy} fuzzy dups removed"
        )
        return unique

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _compute_exact_hash(text: str) -> str:
        """SHA-256 hash of normalized text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _compute_minhash(self, text: str) -> MinHash:
        """
        MinHash signature using character 3-grams.
        """
        mh = MinHash(num_perm=self._num_perm)
        shingles = self._shingle(text, k=3)
        for shingle in shingles:
            mh.update(shingle.encode("utf-8"))
        return mh

    @staticmethod
    def _shingle(text: str, k: int = 3) -> set[str]:
        """
        Character k-grams for robust near-duplicate matching.
        """
        text = " ".join(text.lower().split())
        return {text[i : i + k] for i in range(len(text) - k + 1)}

    def _resolve_conflict(self, incoming: ParsedChunk, existing_keys: list[str]) -> ParsedChunk:
        """
        Select canonical chunk among near-duplicates.
        """
        existing_chunks = [self._lsh_registry[k] for k in existing_keys if k in self._lsh_registry]
        if not existing_chunks:
            return incoming

        strategy = self._keep_strategy
        if strategy == "newest":
            all_candidates = existing_chunks + [incoming]
            return max(
                all_candidates,
                key=lambda c: c.ingestion_ts or "",
            )
        elif strategy == "oldest":
            all_candidates = existing_chunks + [incoming]
            return min(
                all_candidates,
                key=lambda c: c.ingestion_ts or "9999",
            )
        else:
            all_candidates = existing_chunks + [incoming]
            return max(all_candidates, key=lambda c: len(c.text))
