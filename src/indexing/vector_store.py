from __future__ import annotations

import random
from typing import Any

import numpy as np
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.config.settings import get_config, get_secrets
from src.ingestion.parser import ParsedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchResult:
    """
    Search result wrapper (chunk + score + metadata)
    """

    def __init__(
        self,
        chunk: ParsedChunk,
        score: float,
        retrieval_method: str = "hybrid",
        rank: int = 0,
    ) -> None:
        self.chunk = chunk
        self.score = score
        self.retrieval_method = retrieval_method
        self.rank = rank


class DenseVectorStore:
    """
    Qdrant wrapper for dense vector indexing and ANN search.
    """

    def __init__(self) -> None:
        self.cfg = get_config()
        sec = get_secrets()
        self._vs_cfg = self.cfg.vector_store
        self._collection = self._vs_cfg["collection_name"]

        if "localhost" in sec.qdrant_url or "127.0.0.1" in sec.qdrant_url:
            local_path = "qdrant_storage"
            self._client = QdrantClient(path=local_path)
            logger.info(f"Qdrant running in local embedded mode → {local_path}")
        else:
            self._client = QdrantClient(
                url=sec.qdrant_url,
                api_key=sec.qdrant_api_key,
                timeout=30,
            )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection with HNSW parameters if it doesn't exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection in existing:
            logger.debug(
                f"Qdrant collection '{self._collection}' already exists — skipping creation"
            )
            return

        hnsw_cfg = self._vs_cfg["hnsw"]
        dim = self.cfg.embeddings["embedding_dimensions"]

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qdrant_models.VectorParams(
                size=dim,
                distance=qdrant_models.Distance.COSINE,
            ),
            hnsw_config=qdrant_models.HnswConfigDiff(
                m=hnsw_cfg["m"],
                ef_construct=hnsw_cfg["ef_construct"],
                full_scan_threshold=10_000,
            ),
            on_disk_payload=True,
        )
        logger.info(f"Created Qdrant collection '{self._collection}' with HNSW(m={hnsw_cfg['m']})")

    def upsert(self, chunk: ParsedChunk, vector: np.ndarray, namespace: str | None = None) -> None:
        """Upsert a chunk and its embedding into Qdrant."""

        payload = {
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,  # preserve original id for round-trip lookup
            "source_file": chunk.source_file,
            "source_name": chunk.source_name,
            "modality": chunk.modality,
            "language": chunk.language,
            "doc_version": chunk.doc_version,
            "ingestion_ts": chunk.ingestion_ts,
            "namespace": namespace or "default",
            "allowed_roles": chunk.metadata.get("allowed_roles", ["public"]),
            **chunk.metadata,
        }

        payload = self._sanitize_payload(payload)

        self._client.upsert(
            collection_name=self._collection,
            points=[
                qdrant_models.PointStruct(
                    id=self._chunk_id_to_int(chunk.chunk_id),
                    vector=vector.tolist(),
                    payload=payload,
                )
            ],
        )

    @staticmethod
    def _sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Ensure payload is JSON-serializable for Qdrant."""

        _ALLOWED = (str, int, float, bool, type(None))

        def _clean(v: Any) -> Any:
            if isinstance(v, _ALLOWED):
                return v
            if isinstance(v, dict):
                return {str(k): _clean(val) for k, val in v.items()}
            if isinstance(v, list | tuple):
                return [_clean(i) for i in v]
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            # Unknown type (PixelSpace, CoordinatesMetadata, etc.) — stringify it
            return str(v)

        return {str(k): _clean(v) for k, v in payload.items()}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        namespace: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        ef_search: int | None = None,
    ) -> list[SearchResult]:
        """Run ANN search in Qdrant."""

        must_conditions = []

        must_conditions.append(
            qdrant_models.FieldCondition(
                key="hierarchy_level",
                match=qdrant_models.MatchValue(value="paragraph"),
            )
        )

        if namespace and namespace != "default":
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="namespace",
                    match=qdrant_models.MatchValue(value=namespace),
                )
            )

        query_filter = qdrant_models.Filter(must=must_conditions) if must_conditions else None

        search_params = qdrant_models.SearchParams(
            hnsw_ef=ef_search or self._vs_cfg["hnsw"]["ef_search"],
            exact=False,
        )

        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=True,
            with_vectors=False,
        )

        search_results = []

        for hit in response.points:
            payload = hit.payload or {}
            chunk = ParsedChunk(
                text=payload.get("text", ""),
                source_file=payload.get("source_file"),
                source_name=payload.get("source_name"),
                modality=payload.get("modality", "text"),
                language=payload.get("language"),
                doc_version=payload.get("doc_version"),
                ingestion_ts=payload.get("ingestion_ts"),
                metadata=payload,
            )
            chunk.chunk_id = payload.get("chunk_id") or str(hit.id)
            search_results.append(
                SearchResult(chunk=chunk, score=hit.score, retrieval_method="dense")
            )

        return search_results

    @staticmethod
    def _chunk_id_to_int(chunk_id: str | None) -> int:
        """Convert a chunk_id string to a 64-bit integer for use as a Qdrant point ID."""
        if not chunk_id:
            return random.getrandbits(32)
        # Fast path: already a plain decimal integer (legacy round-trip IDs)
        try:
            return int(chunk_id)
        except ValueError:
            pass
        # Hex or UUID: strip hyphens, take first 16 hex chars
        hex_clean = chunk_id.replace("-", "")
        return int(hex_clean[:16], 16)


class SparseIndex:
    """Elasticsearch wrapper for BM25 search."""

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._vs_cfg = cfg.vector_store
        self._index_name = self._vs_cfg["hybrid_search"]["sparse_index_name"]
        self._available = False

        es_kwargs: dict[str, Any] = {"hosts": [sec.elasticsearch_url]}
        if sec.elasticsearch_api_key:
            es_kwargs["api_key"] = sec.elasticsearch_api_key
        self._es = Elasticsearch(**es_kwargs)

        try:
            self._ensure_index()
            self._available = True
        except Exception as exc:
            logger.warning(
                f"Elasticsearch unavailable ({exc}). "
                "Sparse BM25 index disabled — hybrid search will use dense-only retrieval."
            )

    def _ensure_index(self) -> None:
        """Create the Elasticsearch index with optimised BM25 settings if absent."""
        if self._es.indices.exists(index=self._index_name):
            return
        self._es.indices.create(
            index=self._index_name,
            body={
                "settings": {
                    "number_of_shards": 2,
                    "number_of_replicas": 1,
                    "similarity": {
                        "custom_bm25": {
                            "type": "BM25",
                            "k1": 1.2,
                            "b": 0.75,
                        }
                    },
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "similarity": "custom_bm25",
                        },
                        "chunk_id": {"type": "keyword"},
                        "source_name": {"type": "keyword"},
                        "namespace": {"type": "keyword"},
                        "ingestion_ts": {"type": "date"},
                    }
                },
            },
        )
        logger.info(f"Created Elasticsearch index '{self._index_name}'")

    def index_chunk(self, chunk: ParsedChunk) -> None:
        """
        Index a single chunk into Elasticsearch for BM25 retrieval.
        """
        if not self._available:
            return
        doc = {
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "source_name": chunk.source_name,
            "namespace": chunk.metadata.get("namespace", "default"),
            "ingestion_ts": chunk.ingestion_ts,
            **{k: v for k, v in chunk.metadata.items() if isinstance(v, str | int | float | bool)},
        }
        self._es.index(index=self._index_name, id=chunk.chunk_id, document=doc)

    def search(
        self,
        query: str,
        top_k: int = 10,
        namespace: str | None = None,
    ) -> list[SearchResult]:
        """
        BM25 keyword search using Elasticsearch.
        Optionally filtered to a tenant namespace.
        """
        if not self._available:
            return []
        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "text": {
                                    "query": query,
                                    "operator": "or",
                                    "fuzziness": "AUTO",
                                    "prefix_length": 1,
                                    "max_expansions": 50,
                                }
                            }
                        }
                    ],
                    "filter": (
                        [{"term": {"namespace": namespace}}]
                        if namespace and namespace != "default"
                        else []
                    ),
                }
            },
        }
        response = self._es.search(index=self._index_name, body=body)
        results = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            chunk = ParsedChunk(
                text=src.get("text", ""),
                chunk_id=src.get("chunk_id"),
                source_file=src.get("source_file"),
                source_name=src.get("source_name"),
                ingestion_ts=src.get("ingestion_ts"),
                metadata=src,
            )
            results.append(
                SearchResult(chunk=chunk, score=hit["_score"], retrieval_method="sparse")
            )
        return results


class HybridSearchEngine:
    """
    Runs dense and sparse search in parallel and merges results via RRF.
    """

    def __init__(
        self,
        dense_store: DenseVectorStore,
        sparse_index: SparseIndex,
    ) -> None:
        self._dense = dense_store
        self._sparse = sparse_index
        self._cfg = get_config().vector_store["hybrid_search"]

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        namespace: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Runs dense (ANN) and sparse (BM25) search, then fuses results with RRF.
        """
        dense_results = self._dense.search(
            query_vector=query_vector,
            top_k=top_k * 2,
            namespace=namespace,
            metadata_filter=metadata_filter,
        )

        sparse_results = self._sparse.search(
            query=query,
            top_k=top_k * 2,
            namespace=namespace,
        )

        merged = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        return merged

    def _reciprocal_rank_fusion(
        self,
        dense: list[SearchResult],
        sparse: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        rrf_k = self._cfg["rrf_k"]
        scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        def _accumulate(results: list[SearchResult]) -> None:
            for rank, result in enumerate(results, start=1):
                cid = result.chunk.chunk_id or hash(result.chunk.text)
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
                if cid not in chunk_map:
                    chunk_map[cid] = result

        _accumulate(dense)
        _accumulate(sparse)

        sorted_cids = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]
        merged = []
        for rank, cid in enumerate(sorted_cids, start=1):
            result = chunk_map[cid]
            result.score = scores[cid]
            result.retrieval_method = "hybrid"
            result.rank = rank
            merged.append(result)

        return merged
