from __future__ import annotations

import re

import cohere
import numpy as np
import openai
from sentence_transformers import CrossEncoder

from src.config.settings import get_config, get_secrets
from src.indexing.vector_store import (
    DenseVectorStore,
    HybridSearchEngine,
    SearchResult,
)
from src.ingestion.parser import ParsedChunk
from src.query.understanding import ProcessedQuery
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContextItem:
    """
    A single piece of context to be passed to the generation stage.
    Carries the chunk, its final rank, and its source score.
    """

    def __init__(self, chunk: ParsedChunk, score: float, rank: int) -> None:
        self.chunk = chunk
        self.score = score
        self.rank = rank


class Retriever:
    """
    Orchestrates the full retrieval pipeline:
      ANN/hybrid search → parent-child expansion → reranking → context ordering → compression
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        dense_store: DenseVectorStore,
    ) -> None:
        self._search_engine = search_engine
        self._dense_store = dense_store
        cfg = get_config()
        sec = get_secrets()
        self._ret_cfg = cfg.retrieval
        self._ctx_cfg = cfg.retrieval["context_management"]
        self._openai = openai.OpenAI(api_key=sec.openai_api_key)
        self._cohere = cohere.Client(sec.cohere_api_key)

        # Lazy-load local cross-encoder if configured
        self._cross_encoder: CrossEncoder | None = None

    def retrieve(
        self,
        pq: ProcessedQuery,
        query_vector: np.ndarray,
        namespace: str | None = None,
    ) -> list[ContextItem]:
        """
        Full retrieval pipeline for a processed query.

        Args:
            pq:           ProcessedQuery from the query understanding stage.
            query_vector: Dense embedding of the final query.
            namespace:    Tenant namespace for ACL filtering (Challenge 26).

        Returns:
            Ordered list of ContextItems ready for the generation stage.
        """
        top_k_initial = self._ret_cfg["top_k_initial"]
        top_k_final = self._ret_cfg["top_k_final"]

        raw_results: list[SearchResult] = self._search_engine.search(
            query=pq.final_query(),
            query_vector=query_vector,
            top_k=top_k_initial,
            namespace=namespace,
            metadata_filter=None,
        )
        logger.info(f"Initial retrieval: {len(raw_results)} candidates")

        if self._ret_cfg["parent_child"]["enabled"]:
            raw_results = self._expand_to_parents(raw_results)

        if self._ret_cfg["sentence_window"]["enabled"]:
            raw_results = self._expand_sentence_window(raw_results)

        if self._ret_cfg["reranking"]["enabled"]:
            raw_results = self._rerank(pq.final_query(), raw_results)

        raw_results = raw_results[:top_k_final]

        ordered = self._apply_position_aware_ordering(raw_results)

        context_items = self._manage_context(ordered, pq.final_query())

        logger.info(f"Final context: {len(context_items)} items")
        return context_items

    def retrieve_dual(
        self,
        pq: ProcessedQuery,
        query_vector: np.ndarray,
        hyde_vector: np.ndarray,
        namespace: str | None = None,
    ) -> list[ContextItem]:
        """
        Run retrieval twice — once with the raw query vector and once with the
        HyDE hypothetical document vector — then merge via Reciprocal Rank Fusion.
        """
        top_k_initial = self._ret_cfg["top_k_initial"]
        top_k_final = self._ret_cfg["top_k_final"]
        rrf_k = 60

        results_query = self._search_engine.search(
            query=pq.final_query(),
            query_vector=query_vector,
            top_k=top_k_initial,
            namespace=namespace,
            metadata_filter=None,
        )

        results_hyde = self._search_engine.search(
            query=pq.hypothetical_doc or pq.final_query(),
            query_vector=hyde_vector,
            top_k=top_k_initial,
            namespace=namespace,
            metadata_filter=None,
        )

        scores: dict = {}
        chunk_map: dict = {}

        for rank, result in enumerate(results_query, start=1):
            cid = result.chunk.chunk_id or result.chunk.text[:50]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunk_map.setdefault(cid, result)

        for rank, result in enumerate(results_hyde, start=1):
            cid = result.chunk.chunk_id or result.chunk.text[:50]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunk_map.setdefault(cid, result)

        sorted_cids = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k_initial]
        raw_results: list[SearchResult] = []
        for rank, cid in enumerate(sorted_cids, start=1):
            r = chunk_map[cid]
            r.score = scores[cid]
            r.rank = rank
            r.retrieval_method = "hyde_dual"
            raw_results.append(r)

        logger.info(
            f"Dual retrieval merged: {len(results_query)} query results + "
            f"{len(results_hyde)} HyDE results → {len(raw_results)} unique candidates"
        )

        if self._ret_cfg["parent_child"]["enabled"]:
            raw_results = self._expand_to_parents(raw_results)
        if self._ret_cfg["sentence_window"]["enabled"]:
            raw_results = self._expand_sentence_window(raw_results)
        if self._ret_cfg["reranking"]["enabled"]:
            raw_results = self._rerank(pq.final_query(), raw_results)

        raw_results = raw_results[:top_k_final]
        ordered = self._apply_position_aware_ordering(raw_results)
        context_items = self._manage_context(ordered, pq.final_query())

        logger.info(f"Final context (dual): {len(context_items)} items")
        return context_items

    def _expand_to_parents(self, results: list[SearchResult]) -> list[SearchResult]:
        """Replace retrieved child chunks with their parent section chunks."""
        expanded: list[SearchResult] = []
        seen_parent_ids = set()

        for result in results:
            parent_id = result.chunk.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                parent_chunk = self._fetch_chunk_by_id(parent_id)
                if parent_chunk:
                    parent_result = SearchResult(
                        chunk=parent_chunk,
                        score=result.score,
                        retrieval_method=result.retrieval_method,
                        rank=result.rank,
                    )
                    expanded.append(parent_result)
                    seen_parent_ids.add(parent_id)
                    continue
            expanded.append(result)
        return expanded

    def _fetch_chunk_by_id(self, chunk_id: str) -> ParsedChunk | None:
        """
        Retrieve a single chunk from Qdrant by its point ID.
        Used for parent-child lookup.
        """
        try:
            points = self._dense_store._client.retrieve(
                collection_name=self._dense_store._collection,
                ids=[DenseVectorStore._chunk_id_to_int(chunk_id)],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None
            payload = points[0].payload or {}
            chunk = ParsedChunk(
                text=payload.get("text", ""),
                source_file=payload.get("source_file"),
                source_name=payload.get("source_name"),
                modality=payload.get("modality", "text"),
                metadata=payload,
            )
            chunk.chunk_id = chunk_id
            return chunk
        except Exception as exc:
            logger.warning(f"Parent fetch failed for {chunk_id}: {exc}")
            return None

    def _expand_sentence_window(self, results: list[SearchResult]) -> list[SearchResult]:
        """Expand each retrieved chunk with surrounding sentences from the original document."""
        for result in results:
            surrounding = result.chunk.metadata.get("surrounding_text")
            if surrounding:
                result.chunk.text = surrounding
        return results

    def _rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """
        Apply cross-encoder re-ranking to the initial retrieval candidates.
        Supports Cohere, ColBERT, BGE, and LLM-based rerankers.
        """
        model_choice = self._ret_cfg["reranking"]["model"]
        documents = [r.chunk.text for r in results]

        if model_choice == "cohere":
            return self._rerank_cohere(query, documents, results)
        elif model_choice == "bge":
            return self._rerank_cross_encoder(query, documents, results)
        elif model_choice == "llm_pointwise":
            return self._rerank_llm(query, documents, results)
        else:
            logger.warning(f"Unknown reranker '{model_choice}' — skipping reranking")
            return results

    def _rerank_cohere(
        self, query: str, documents: list[str], results: list[SearchResult]
    ) -> list[SearchResult]:
        """Re-rank using the Cohere cross-encoder reranker API."""
        model = self._ret_cfg["reranking"]["cohere_model"]
        response = self._cohere.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=len(documents),
        )
        reranked: list[SearchResult] = []
        for i, rerank_result in enumerate(response.results):
            original = results[rerank_result.index]
            original.score = rerank_result.relevance_score
            original.rank = i + 1
            reranked.append(original)
        return reranked

    def _rerank_cross_encoder(
        self, query: str, documents: list[str], results: list[SearchResult]
    ) -> list[SearchResult]:
        """Re-rank using a local BGE cross-encoder model."""
        if self._cross_encoder is None:
            model_name = self._ret_cfg["reranking"]["bge_model"]
            self._cross_encoder = CrossEncoder(model_name)
        pairs = [(query, doc) for doc in documents]
        scores = self._cross_encoder.predict(pairs)
        for result, score in zip(results, scores, strict=False):
            result.score = float(score)
        results.sort(key=lambda r: r.score, reverse=True)
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        return results

    def _rerank_llm(
        self, query: str, documents: list[str], results: list[SearchResult]
    ) -> list[SearchResult]:
        """
        Re-rank using an LLM with a listwise scoring prompt.
        Most expensive but highest quality; use only for critical queries.
        """
        model = self._ret_cfg["reranking"]["llm_rerank_model"]
        doc_list_str = "\n".join(f"[{i+1}] {doc[:500]}" for i, doc in enumerate(documents))
        prompt = (
            f"Query: {query}\n\n"
            f"Documents:\n{doc_list_str}\n\n"
            "Rank the documents by relevance to the query. "
            "Return ONLY a comma-separated list of document numbers in order from most to least relevant. "
            "Example: 3, 1, 5, 2, 4"
        )
        response = self._openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        raw = response.choices[0].message.content.strip()
        try:
            order = [int(x.strip()) - 1 for x in raw.split(",")]
            reranked = [results[i] for i in order if 0 <= i < len(results)]
            mentioned = set(order)
            reranked += [r for i, r in enumerate(results) if i not in mentioned]
            for rank, result in enumerate(reranked, start=1):
                result.rank = rank
            return reranked
        except (ValueError, IndexError):
            logger.warning("LLM reranker returned unparseable output — using original order")
            return results

    def _apply_position_aware_ordering(self, results: list[SearchResult]) -> list[SearchResult]:
        """Reorder chunks so the most relevant appear at the start and end of the context window."""
        ordering = self._ret_cfg["context_ordering"]["strategy"]

        if ordering != "position_aware" or len(results) < 3:
            return results

        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        if not sorted_results:
            return results

        best = sorted_results[0]
        second_best = sorted_results[1] if len(sorted_results) > 1 else None
        middle = sorted_results[2:]

        ordered = [best] + middle
        if second_best:
            ordered.append(second_best)

        logger.debug(f"Position-aware ordering applied to {len(ordered)} chunks")
        return ordered

    def _manage_context(self, results: list[SearchResult], query: str) -> list[ContextItem]:
        """
        Ensure the total context fits within the model's context window.
        Applies compression if necessary.
        """
        max_tokens = self._ctx_cfg["max_context_tokens"]
        strategy = self._ctx_cfg["compression_model"]

        context_items: list[ContextItem] = []
        total_tokens = 0

        for rank, result in enumerate(results, start=1):
            chunk_text = result.chunk.text

            estimated_tokens = len(chunk_text) // 4

            if total_tokens + estimated_tokens > max_tokens:
                if strategy == "llm_lingua":
                    chunk_text = self._compress_llm_lingua(chunk_text)
                elif strategy == "extractive":
                    chunk_text = self._compress_extractive(chunk_text, max_sentences=3)
                estimated_tokens = len(chunk_text) // 4
                if total_tokens + estimated_tokens > max_tokens:
                    logger.warning(
                        f"Context budget exhausted at rank {rank} — stopping context assembly"
                    )
                    break

            result.chunk.text = chunk_text
            total_tokens += estimated_tokens
            context_items.append(ContextItem(chunk=result.chunk, score=result.score, rank=rank))

        return context_items

    def _compress_llm_lingua(self, text: str) -> str:
        """Compress a text chunk using an LLM to retain key information."""
        ratio = self._ctx_cfg["llm_lingua_ratio"]
        target_length = int(len(text) * ratio)
        prompt = (
            f"Compress the following text to approximately {target_length} characters "
            f"while preserving all key facts and information. Return ONLY the compressed text.\n\n"
            f"{text}"
        )
        response = self._openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=target_length // 4 + 50,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _compress_extractive(text: str, max_sentences: int = 3) -> str:
        """Extractive compression: keep only the first N sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:max_sentences])
