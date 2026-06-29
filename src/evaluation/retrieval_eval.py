"""
Retrieval quality evaluation against labeled queries: Precision@K, Recall@K, MRR, Hit Rate, NDCG.

Metrics
-------
- Precision@K   : fraction of top-K results that are relevant
- Recall@K      : fraction of all relevant items found in top-K
- MRR           : reciprocal rank of the first relevant result
- Hit Rate@K    : fraction of queries with at least one relevant result in top-K
- NDCG@K        : normalised discounted cumulative gain
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from src.indexing.embedder import QueryEmbedder
from src.query.understanding import ProcessedQuery
from src.retrieval.retriever import ContextItem, Retriever
from src.utils.logger import get_logger

try:
    from cohere.errors.too_many_requests_error import TooManyRequestsError
except ImportError:  # pragma: no cover - cohere optional / version dependent
    TooManyRequestsError = None

logger = get_logger(__name__)


@dataclass
class LabeledQuery:
    """A single labeled evaluation example."""

    query: str
    relevant_chunk_ids: list[str]
    namespace: str = "default"
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryRetrievalResult:
    """Per-query retrieval metrics."""

    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit: bool
    ndcg: float


@dataclass
class RetrievalEvalReport:
    """Aggregate retrieval evaluation results across all queries."""

    k: int
    num_queries: int
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    hit_rate: float
    mean_ndcg: float
    per_query: list[QueryRetrievalResult] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary table."""
        lines = [
            "",
            "=" * 52,
            f"  RETRIEVAL EVALUATION  (K={self.k}, N={self.num_queries})",
            "=" * 52,
            f"  Precision@{self.k:<3}  {self.mean_precision:.4f}",
            f"  Recall@{self.k:<6}  {self.mean_recall:.4f}",
            f"  MRR          {self.mean_mrr:.4f}",
            f"  Hit Rate@{self.k:<3}  {self.hit_rate:.4f}",
            f"  NDCG@{self.k:<6}  {self.mean_ndcg:.4f}",
            "=" * 52,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "num_queries": self.num_queries,
            f"precision@{self.k}": round(self.mean_precision, 4),
            f"recall@{self.k}": round(self.mean_recall, 4),
            "mrr": round(self.mean_mrr, 4),
            f"hit_rate@{self.k}": round(self.hit_rate, 4),
            f"ndcg@{self.k}": round(self.mean_ndcg, 4),
        }


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of top-K retrieved items that are relevant."""
    if k == 0:
        return 0.0
    top_k = list(retrieved)[:k]
    relevant_set = set(relevant)
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of all relevant items found in the top-K results."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(retrieved)[:k]
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant result (0 if none found)."""
    relevant_set = set(relevant)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K."""
    relevant_set = set(relevant)
    top_k = list(retrieved)[:k]

    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, cid in enumerate(top_k, start=1) if cid in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


class RetrievalEvaluator:

    def __init__(
        self,
        embedder: QueryEmbedder,
        retriever: Retriever,
        request_delay: float = 0.0,
        max_retries: int = 2,
        retry_backoff: float = 20.0,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._request_delay = request_delay
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def _retrieve_ids(
        self,
        query: str,
        namespace: str,
        k: int,
    ) -> list[str]:
        """Run retrieval and return chunk IDs in rank order."""
        query_vector: np.ndarray = self._embedder.embed_query(query)

        pq = ProcessedQuery(
            original_query=query,
            standalone_query=query,
            sub_questions=[],
            metadata_filters={},
        )

        if self._request_delay > 0:
            time.sleep(self._request_delay)

        attempt = 0
        while True:
            try:
                context_items: list[ContextItem] = self._retriever.retrieve(
                    pq=pq,
                    query_vector=query_vector,
                    namespace=namespace,
                )
                break
            except Exception as exc:
                is_rate_limit = (
                    (TooManyRequestsError is not None and isinstance(exc, TooManyRequestsError))
                    or "429" in str(exc)
                    or "too many requests" in str(exc).lower()
                )

                if is_rate_limit and attempt < self._max_retries:
                    attempt += 1
                    logger.warning(
                        "Rate limited (attempt %d/%d) — sleeping %.0fs before retry: %s",
                        attempt,
                        self._max_retries,
                        self._retry_backoff,
                        exc,
                    )
                    time.sleep(self._retry_backoff)
                    continue
                raise

        return [
            item.chunk.chunk_id for item in context_items[:k] if item.chunk.chunk_id is not None
        ]

    def evaluate_query(
        self,
        labeled: LabeledQuery,
        k: int,
    ) -> QueryRetrievalResult:
        """Evaluate retrieval for a single labeled query."""
        try:
            retrieved_ids = self._retrieve_ids(labeled.query, labeled.namespace, k)
        except Exception as exc:
            logger.error(
                "Retrieval failed for query '%s': %s",
                labeled.query[:80],
                exc,
                exc_info=True,
            )
            retrieved_ids = []

        return QueryRetrievalResult(
            query=labeled.query,
            retrieved_ids=retrieved_ids,
            relevant_ids=labeled.relevant_chunk_ids,
            precision_at_k=precision_at_k(retrieved_ids, labeled.relevant_chunk_ids, k),
            recall_at_k=recall_at_k(retrieved_ids, labeled.relevant_chunk_ids, k),
            mrr=mean_reciprocal_rank(retrieved_ids, labeled.relevant_chunk_ids),
            hit=any(cid in set(labeled.relevant_chunk_ids) for cid in retrieved_ids[:k]),
            ndcg=ndcg_at_k(retrieved_ids, labeled.relevant_chunk_ids, k),
        )

    def evaluate(
        self,
        labeled_queries: list[LabeledQuery],
        k: int = 10,
    ) -> RetrievalEvalReport:
        """Evaluate retrieval across an entire labeled dataset."""
        if not labeled_queries:
            raise ValueError("labeled_queries must not be empty.")

        logger.info(
            "Starting retrieval evaluation: %d queries, K=%d",
            len(labeled_queries),
            k,
        )

        per_query: list[QueryRetrievalResult] = []
        for i, labeled in enumerate(labeled_queries):
            result = self.evaluate_query(labeled, k)
            per_query.append(result)
            logger.debug(
                "[%d/%d] '%s' — P@%d=%.3f R@%d=%.3f MRR=%.3f",
                i + 1,
                len(labeled_queries),
                labeled.query[:60],
                k,
                result.precision_at_k,
                k,
                result.recall_at_k,
                result.mrr,
            )

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        report = RetrievalEvalReport(
            k=k,
            num_queries=len(per_query),
            mean_precision=_mean([r.precision_at_k for r in per_query]),
            mean_recall=_mean([r.recall_at_k for r in per_query]),
            mean_mrr=_mean([r.mrr for r in per_query]),
            hit_rate=_mean([float(r.hit) for r in per_query]),
            mean_ndcg=_mean([r.ndcg for r in per_query]),
            per_query=per_query,
        )

        logger.info(report.summary())
        return report

    def evaluate_baseline(
        self,
        labeled_queries: list[LabeledQuery],
        k: int = 10,
    ) -> RetrievalEvalReport:
        """Evaluate a dense-only baseline (no reranking, no graph) for pipeline uplift comparison."""
        logger.info("Running baseline (dense-only) retrieval evaluation.")

        per_query: list[QueryRetrievalResult] = []
        for labeled in labeled_queries:
            try:
                query_vector = self._embedder.embed_query(labeled.query)
                results = self._retriever._dense_store.search(
                    query_vector=query_vector,
                    top_k=k,
                    namespace=labeled.namespace,
                )
                retrieved_ids = [r.chunk.chunk_id for r in results if r.chunk.chunk_id][:k]
            except Exception as exc:
                logger.warning("Baseline retrieval failed for '%s': %s", labeled.query[:60], exc)
                retrieved_ids = []

            per_query.append(
                QueryRetrievalResult(
                    query=labeled.query,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=labeled.relevant_chunk_ids,
                    precision_at_k=precision_at_k(retrieved_ids, labeled.relevant_chunk_ids, k),
                    recall_at_k=recall_at_k(retrieved_ids, labeled.relevant_chunk_ids, k),
                    mrr=mean_reciprocal_rank(retrieved_ids, labeled.relevant_chunk_ids),
                    hit=any(cid in set(labeled.relevant_chunk_ids) for cid in retrieved_ids[:k]),
                    ndcg=ndcg_at_k(retrieved_ids, labeled.relevant_chunk_ids, k),
                )
            )

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        report = RetrievalEvalReport(
            k=k,
            num_queries=len(per_query),
            mean_precision=_mean([r.precision_at_k for r in per_query]),
            mean_recall=_mean([r.recall_at_k for r in per_query]),
            mean_mrr=_mean([r.mrr for r in per_query]),
            hit_rate=_mean([float(r.hit) for r in per_query]),
            mean_ndcg=_mean([r.ndcg for r in per_query]),
            per_query=per_query,
        )

        logger.info("BASELINE %s", report.summary())
        return report
