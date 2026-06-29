from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config.settings import get_config
from src.ingestion.chunker import ChunkNode
from src.ingestion.parser import ParsedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingRouter:
    """
    Routes chunks to embedding models based on domain and language.
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._emb_cfg = self._cfg.embeddings

        self._default_model_name: str = self._emb_cfg["default_model"]
        self._multilingual_model_name: str = self._emb_cfg["multilingual_model"]
        self._domain_model_map: dict[str, str] = self._emb_cfg.get("domain_models", {})

        self._batch_size: int = self._emb_cfg["batch_size"]
        self._lang_detection: bool = self._emb_cfg["language_detection"]

        self._model_cache: dict[str, SentenceTransformer] = {}
        self._load_model(self._default_model_name)

    def embed_nodes(self, nodes: list[ChunkNode]) -> list[tuple[ChunkNode, np.ndarray]]:
        """Embed a list of nodes, routing each to the appropriate model by domain and language."""

        groups: dict[str, list[tuple[int, ChunkNode]]] = {}

        for idx, node in enumerate(nodes):
            model_name = self._select_model(node.chunk)
            groups.setdefault(model_name, []).append((idx, node))

        result_pairs: list[tuple[ChunkNode, np.ndarray] | None] = [None] * len(nodes)

        for model_name, indexed_nodes in groups.items():
            model = self._load_model(model_name)
            texts = [n.chunk.text for _, n in indexed_nodes]

            vectors = self._batch_encode(model, texts)

            for i, (original_idx, node) in enumerate(indexed_nodes):
                result_pairs[original_idx] = (node, vectors[i])

        return [(node, vec) for pair in result_pairs if pair for node, vec in [pair]]

    def _select_model(self, chunk: ParsedChunk) -> str:
        """Select embedding model based on domain and language."""

        source = (chunk.source_name or "").lower()
        for domain_key, domain_model in self._domain_model_map.items():
            if domain_key.lower() in source:
                logger.debug(f"Domain model '{domain_model}' selected for source '{source}'")
                return domain_model

        if self._lang_detection and chunk.language and chunk.language != "en":
            logger.debug(f"Multilingual model selected for language '{chunk.language}'")
            return self._multilingual_model_name

        return self._default_model_name

    def _load_model(self, model_name: str) -> SentenceTransformer:
        """Load and cache SentenceTransformer model."""
        if model_name not in self._model_cache:
            logger.info(f"Loading embedding model: {model_name}")
            self._model_cache[model_name] = SentenceTransformer(
                model_name,
                device="cuda" if self._gpu_available() else "cpu",
            )
        return self._model_cache[model_name]

    def _batch_encode(self, model: SentenceTransformer, texts: list[str]) -> np.ndarray:
        """Batch-encode texts for efficiency."""
        all_vectors: list[np.ndarray] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            all_vectors.append(vectors)

        return np.vstack(all_vectors)

    @staticmethod
    def _gpu_available() -> bool:
        """Check CUDA availability."""
        try:
            return torch.cuda.is_available()
        except ImportError:
            return False


class QueryEmbedder:
    """
    Embeds queries using the shared embedding router.
    """

    def __init__(self, router: EmbeddingRouter | None = None) -> None:
        self._router = router or EmbeddingRouter()
        self._cfg = get_config()
        self._default_model_name = self._cfg.embeddings["default_model"]

    def embed_query(self, query: str, language: str | None = None) -> np.ndarray:
        """Embed a single query string."""
        dummy_chunk = ParsedChunk(text=query, language=language)
        model_name = self._router._select_model(dummy_chunk)
        model = self._router._load_model(model_name)
        vector: np.ndarray = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector
