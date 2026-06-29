from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.agents.orchestrator import RAGOrchestrator
from src.config.settings import get_config
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.retrieval_eval import RetrievalEvaluator
from src.generation.generator import Generator
from src.indexing.embedder import EmbeddingRouter, QueryEmbedder
from src.indexing.vector_store import DenseVectorStore, HybridSearchEngine, SparseIndex
from src.operations.ops_middleware import AccessControlMiddleware, PIIGuard
from src.query.understanding import QueryUnderstanding
from src.retrieval.retriever import Retriever

try:
    from src.graphrag.graph_retriever import GraphRetriever
    from src.graphrag.neo4j_store import Neo4jGraphStore

    _GRAPHRAG_AVAILABLE = True
except ImportError:
    Neo4jGraphStore = None
    GraphRetriever = None
    _GRAPHRAG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AppContainer:
    """Holds every application-level singleton."""

    embedder: object = field(default=None, repr=False)
    dense_store: object = field(default=None, repr=False)
    sparse_index: object = field(default=None, repr=False)
    search_engine: object = field(default=None, repr=False)
    retriever: object = field(default=None, repr=False)
    generator: object = field(default=None, repr=False)
    evaluator: object = field(default=None, repr=False)
    pii_guard: object = field(default=None, repr=False)
    acl: object = field(default=None, repr=False)
    query_understanding: object = field(default=None, repr=False)
    retrieval_evaluator: object = field(default=None, repr=False)
    graph_retriever: object | None = field(default=None, repr=False)
    orchestrator: object = field(default=None, repr=False)

    _started: bool = False

    def startup(self) -> None:
        """
        Initialise every singleton in dependency order.
        Must be called exactly once, inside the FastAPI lifespan.
        """
        if self._started:
            return

        logger.info("AppContainer.startup — initialising retrieval stack")

        self.acl = AccessControlMiddleware()
        self.pii_guard = PIIGuard()

        router = EmbeddingRouter()
        self.embedder = QueryEmbedder(router=router)
        logger.info("QueryEmbedder ready")

        self.dense_store = DenseVectorStore()
        self.sparse_index = SparseIndex()
        self.search_engine = HybridSearchEngine(self.dense_store, self.sparse_index)
        logger.info(
            "Vector stores ready (dense=%s, sparse_available=%s)",
            type(self.dense_store).__name__,
            getattr(self.sparse_index, "_available", False),
        )

        self.retriever = Retriever(self.search_engine, self.dense_store)
        logger.info("Retriever ready")

        self.generator = Generator()
        self.evaluator = RAGEvaluator()
        self.retrieval_evaluator = RetrievalEvaluator(
            embedder=self.embedder, retriever=self.retriever
        )
        logger.info("Generator and evaluator ready")

        self.query_understanding = QueryUnderstanding()
        logger.info("QueryUnderstanding ready")

        self.graph_retriever = self._try_init_graph_retriever()

        self.orchestrator = RAGOrchestrator(container=self)
        logger.info("RAGOrchestrator ready")

        self._started = True
        logger.info("AppContainer.startup complete")

    def shutdown(self) -> None:
        """Close connections gracefully."""
        if not self._started:
            return
        logger.info("AppContainer.shutdown — releasing resources")
        try:
            if self.dense_store and hasattr(self.dense_store, "_client"):
                self.dense_store._client.close()
        except Exception as exc:
            logger.warning("DenseVectorStore close error (non-fatal): %s", exc)
        self._started = False

    def _try_init_graph_retriever(self) -> object | None:
        cfg = get_config()
        gr_cfg = cfg.graphrag
        if not gr_cfg.get("enabled", False):
            logger.info("GraphRAG disabled — skipping GraphRetriever init")
            return None
        if not _GRAPHRAG_AVAILABLE:
            logger.warning("GraphRAG enabled in config but dependencies not installed.")
            return None
        try:
            graph_store = Neo4jGraphStore()
            gr = GraphRetriever(
                graph_store=graph_store,
                vector_retriever=self.retriever,
                search_engine=self.search_engine,
                dense_store=self.dense_store,
            )
            logger.info("GraphRetriever ready")
            return gr
        except Exception as exc:
            logger.error("GraphRetriever init failed (non-fatal): %s", exc)
            return None


_container: AppContainer | None = None


def get_container() -> AppContainer:
    """Return the application container. Raises if startup() was not called."""
    if _container is None or not _container._started:
        raise RuntimeError(
            "AppContainer not initialised. "
            "Call src.core.container.init_container() inside the FastAPI lifespan."
        )
    return _container


def init_container() -> AppContainer:
    """Create and start the container. Called once from app lifespan."""
    global _container
    _container = AppContainer()
    _container.startup()
    return _container
