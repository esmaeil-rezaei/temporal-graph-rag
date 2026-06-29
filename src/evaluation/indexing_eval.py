"""Indexing quality evaluation: chunk coherence, embedding spot-checks, and entity coverage."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field

import numpy as np

from src.indexing.embedder import QueryEmbedder
from src.ingestion.parser import ParsedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkCoherenceResult:
    chunk_id: str
    text_preview: str
    coherence_score: float
    reason: str


@dataclass
class EmbeddingPairResult:
    chunk_id_a: str
    chunk_id_b: str
    expected_relation: str
    cosine_similarity: float
    passed: bool
    note: str


@dataclass
class EntityCoverageResult:
    entity: str
    found_in_chunks: list[str]  # chunk IDs that contain the entity
    covered: bool


@dataclass
class IndexingEvalReport:
    num_chunks_sampled: int
    mean_coherence: float
    low_coherence_chunks: list[ChunkCoherenceResult]
    all_coherence: list[ChunkCoherenceResult] = field(default_factory=list)
    embedding_pass_rate: float = 0.0
    embedding_results: list[EmbeddingPairResult] = field(default_factory=list)
    entity_coverage_rate: float = 0.0
    entity_results: list[EntityCoverageResult] = field(default_factory=list)

    def summary(self) -> str:
        low = len(self.low_coherence_chunks)
        lines = [
            "",
            "=" * 52,
            f"  INDEXING EVALUATION  (N={self.num_chunks_sampled})",
            "=" * 52,
            f"  Chunk Coherence (mean)   {self.mean_coherence:.3f}",
            f"  Low-Coherence Chunks     {low}",
            f"  Embedding Pass Rate      {self.embedding_pass_rate:.3f}",
            f"  Entity Coverage Rate     {self.entity_coverage_rate:.3f}",
            "=" * 52,
        ]
        if self.low_coherence_chunks:
            lines.append("  ⚠  Low-coherence samples:")
            for r in self.low_coherence_chunks[:5]:
                lines.append(f"     [{r.chunk_id}] {r.text_preview[:60]}…")
                lines.append(f"       score={r.coherence_score:.2f} — {r.reason}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "num_chunks_sampled": self.num_chunks_sampled,
            "mean_coherence": round(self.mean_coherence, 4),
            "low_coherence_count": len(self.low_coherence_chunks),
            "embedding_pass_rate": round(self.embedding_pass_rate, 4),
            "entity_coverage_rate": round(self.entity_coverage_rate, 4),
        }


class IndexingEvaluator:

    def __init__(
        self,
        embedder: QueryEmbedder,
        openai_client,
        judge_model: str = "gpt-4-turbo",
        coherence_threshold: float = 0.6,
    ) -> None:
        self._embedder = embedder
        self._client = openai_client
        self._judge_model = judge_model
        self._coherence_threshold = coherence_threshold

    def _score_chunk_coherence(self, chunk: ParsedChunk) -> ChunkCoherenceResult:
        """Ask the LLM judge whether a single chunk makes sense in isolation."""
        prompt = (
            "You are evaluating text chunk quality for a retrieval-augmented system.\n\n"
            "Assess the following chunk. A good chunk:\n"
            "  - Is a complete, self-contained thought (no abrupt mid-sentence start/end)\n"
            "  - Does not require external context to be understandable\n"
            "  - Is long enough to convey meaningful information (≥ 40 words preferred)\n\n"
            f'Chunk:\n"""\n{chunk.text[:800]}\n"""\n\n'
            "Return ONLY valid JSON with exactly two keys:\n"
            '{"score": <float 0.0-1.0>, "reason": "<one sentence>"}'
        )

        try:
            response = self._client.chat.completions.create(
                model=self._judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            score = max(0.0, min(1.0, float(parsed["score"])))
            reason = str(parsed.get("reason", ""))
        except Exception as exc:
            logger.warning(
                "Coherence judge failed for chunk '%s': %s",
                chunk.chunk_id,
                exc,
            )
            score = 0.5
            reason = f"Judge error: {exc}"

        return ChunkCoherenceResult(
            chunk_id=chunk.chunk_id or "",
            text_preview=chunk.text[:120],
            coherence_score=score,
            reason=reason,
        )

    def evaluate_chunk_coherence(
        self,
        chunks: list[ParsedChunk],
        sample_size: int | None = None,
    ) -> list[ChunkCoherenceResult]:
        """Score coherence for a list of chunks (optionally sampled)."""
        import random

        population = chunks
        if sample_size and len(chunks) > sample_size:
            population = random.sample(chunks, sample_size)

        results: list[ChunkCoherenceResult] = []
        for i, chunk in enumerate(population):
            result = self._score_chunk_coherence(chunk)
            results.append(result)
            logger.debug(
                "[%d/%d] chunk='%s' coherence=%.2f",
                i + 1,
                len(population),
                chunk.chunk_id,
                result.coherence_score,
            )

        return results

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def evaluate_embedding_quality(
        self,
        related_pairs: list[tuple[ParsedChunk, ParsedChunk]],
        unrelated_pairs: list[tuple[ParsedChunk, ParsedChunk]],
        related_threshold: float = 0.65,
        unrelated_ceiling: float = 0.35,
    ) -> list[EmbeddingPairResult]:
        """Spot-check embedding similarity on known-related and known-unrelated chunk pairs."""
        results: list[EmbeddingPairResult] = []

        for chunk_a, chunk_b in related_pairs:
            vec_a = self._embedder.embed_query(chunk_a.text[:512])
            vec_b = self._embedder.embed_query(chunk_b.text[:512])
            sim = self._cosine_similarity(vec_a, vec_b)
            passed = sim >= related_threshold
            results.append(
                EmbeddingPairResult(
                    chunk_id_a=chunk_a.chunk_id or "",
                    chunk_id_b=chunk_b.chunk_id or "",
                    expected_relation="related",
                    cosine_similarity=round(sim, 4),
                    passed=passed,
                    note=(
                        f"OK (sim={sim:.3f} ≥ {related_threshold})"
                        if passed
                        else f"FAIL: sim={sim:.3f} < threshold {related_threshold}"
                    ),
                )
            )

        for chunk_a, chunk_b in unrelated_pairs:
            vec_a = self._embedder.embed_query(chunk_a.text[:512])
            vec_b = self._embedder.embed_query(chunk_b.text[:512])
            sim = self._cosine_similarity(vec_a, vec_b)
            passed = sim <= unrelated_ceiling
            results.append(
                EmbeddingPairResult(
                    chunk_id_a=chunk_a.chunk_id or "",
                    chunk_id_b=chunk_b.chunk_id or "",
                    expected_relation="unrelated",
                    cosine_similarity=round(sim, 4),
                    passed=passed,
                    note=(
                        f"OK (sim={sim:.3f} ≤ {unrelated_ceiling})"
                        if passed
                        else f"FAIL: sim={sim:.3f} > ceiling {unrelated_ceiling}"
                    ),
                )
            )

        pass_count = sum(1 for r in results if r.passed)
        logger.info(
            "Embedding quality: %d/%d pairs passed",
            pass_count,
            len(results),
        )
        return results

    def evaluate_entity_coverage(
        self,
        chunks: list[ParsedChunk],
        expected_entities: list[str],
        case_sensitive: bool = False,
    ) -> list[EntityCoverageResult]:
        """Check that each expected entity appears in at least one chunk."""
        results: list[EntityCoverageResult] = []

        for entity in expected_entities:
            needle = entity if case_sensitive else entity.lower()
            found_in: list[str] = []

            for chunk in chunks:
                haystack = chunk.text if case_sensitive else chunk.text.lower()
                if needle in haystack:
                    found_in.append(chunk.chunk_id or "")

            results.append(
                EntityCoverageResult(
                    entity=entity,
                    found_in_chunks=found_in,
                    covered=bool(found_in),
                )
            )

        covered = sum(1 for r in results if r.covered)
        logger.info(
            "Entity coverage: %d/%d entities found in chunks",
            covered,
            len(expected_entities),
        )
        return results

    def evaluate(
        self,
        chunks: list[ParsedChunk],
        coherence_sample_size: int | None = 50,
        related_pairs: list[tuple[ParsedChunk, ParsedChunk]] | None = None,
        unrelated_pairs: list[tuple[ParsedChunk, ParsedChunk]] | None = None,
        expected_entities: list[str] | None = None,
    ) -> IndexingEvalReport:
        """Run all three indexing checks (coherence, embedding quality, entity coverage)."""
        logger.info("Starting indexing evaluation on %d chunks.", len(chunks))

        coherence_results = self.evaluate_chunk_coherence(chunks, coherence_sample_size)
        scores = [r.coherence_score for r in coherence_results]
        mean_coherence = statistics.mean(scores) if scores else 0.0
        low_coh = [r for r in coherence_results if r.coherence_score < self._coherence_threshold]

        emb_results: list[EmbeddingPairResult] = []
        emb_pass_rate = 1.0
        if related_pairs or unrelated_pairs:
            emb_results = self.evaluate_embedding_quality(
                related_pairs or [],
                unrelated_pairs or [],
            )
            if emb_results:
                emb_pass_rate = sum(1 for r in emb_results if r.passed) / len(emb_results)

        ent_results: list[EntityCoverageResult] = []
        ent_coverage_rate = 1.0
        if expected_entities:
            ent_results = self.evaluate_entity_coverage(chunks, expected_entities)
            if ent_results:
                ent_coverage_rate = sum(1 for r in ent_results if r.covered) / len(ent_results)

        report = IndexingEvalReport(
            num_chunks_sampled=len(coherence_results),
            mean_coherence=mean_coherence,
            low_coherence_chunks=low_coh,
            all_coherence=coherence_results,
            embedding_pass_rate=emb_pass_rate,
            embedding_results=emb_results,
            entity_coverage_rate=ent_coverage_rate,
            entity_results=ent_results,
        )

        logger.info(report.summary())
        return report
