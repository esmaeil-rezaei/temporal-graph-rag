from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import get_config
from src.indexing.embedder import EmbeddingRouter
from src.indexing.vector_store import DenseVectorStore, SparseIndex
from src.ingestion.chunker import ChunkNode, TextChunker
from src.ingestion.consolidator import ChunkConsolidator
from src.ingestion.deduplicator import Deduplicator
from src.ingestion.graph_handler import GraphIngestionHandler
from src.ingestion.parser import DocumentParser, ParsedChunk
from src.operations.ops_middleware import PIIGuard
from src.utils.file_utils import extract_version, hash_file
from src.utils.logger import get_logger, set_correlation_id

try:
    from src.graphrag.community import CommunityDetector
    from src.graphrag.extractor import GraphExtractor
    from src.graphrag.neo4j_store import Neo4jGraphStore

    _GRAPHRAG_AVAILABLE = True
except ImportError as _graphrag_import_err:
    _GRAPHRAG_AVAILABLE = False
    GraphExtractor = None
    Neo4jGraphStore = None
    CommunityDetector = None

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    scan → parse → stamp → PII redact → fingerprint
           → consolidate → dedup → chunk → embed → upsert
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._kb_cfg = self._cfg.knowledge_base
        self._ingest_cfg = self._cfg.ingestion
        self._versioning_cfg = self._ingest_cfg["versioning"]

        self._parser = DocumentParser()
        self._pii_guard = PIIGuard()
        self._consolidator = ChunkConsolidator()
        self._deduplicator = Deduplicator()
        self._chunker = TextChunker()
        self._embedder = EmbeddingRouter()
        self._dense_store = DenseVectorStore()
        self._sparse_index = SparseIndex()

        self._processed_hashes: dict[str, str] = {}

        self._graph_enabled: bool = False
        self._graph_handler: GraphIngestionHandler | None = None
        self._graph_store = None
        self._community_detector = None
        self._run_community_detection: bool = False

        self._init_graphrag()

    def _init_graphrag(self) -> None:
        """Initialise the optional GraphRAG layer (Neo4j + LLM extraction)."""
        gr_cfg = self._cfg.get("graphrag", {})

        if not gr_cfg.get("enabled", False):
            return

        if not _GRAPHRAG_AVAILABLE:
            logger.warning(
                "graphrag.enabled=true in config but GraphRAG dependencies "
                "are not installed. Run: pip install neo4j python-louvain cdlib"
            )
            return

        try:
            self._graph_store = Neo4jGraphStore()
            graph_extractor = GraphExtractor()
            self._graph_handler = GraphIngestionHandler(graph_extractor, self._graph_store)
            self._graph_enabled = True
            logger.info("GraphRAG pipeline enabled (Neo4j + LLM extraction)")

            if gr_cfg.get("run_community_detection", False):
                self._community_detector = CommunityDetector(self._graph_store)
                self._run_community_detection = True
                logger.info("Community detection enabled — will run after ingestion")

        except Exception as exc:
            logger.error(
                "GraphRAG initialisation failed — continuing without graph layer: %s",
                exc,
            )

    def run(self, namespace: str | None = None) -> dict[str, int]:
        """Scan the knowledge base and ingest all supported files."""
        set_correlation_id()
        root = Path(self._kb_cfg["root_dir"])

        if namespace:
            root = root / namespace

        if not root.exists():
            raise FileNotFoundError(f"Knowledge base root not found: {root}")

        parsing_cfg = self._ingest_cfg["parsing"]
        original_img_dir = parsing_cfg["image_output_dir"]
        if namespace:
            parsing_cfg["image_output_dir"] = f"{original_img_dir}/{namespace}"

        supported_ext: set[str] = set(self._kb_cfg["supported_extensions"])
        files = [f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in supported_ext]
        logger.info(f"Ingestion started: {len(files)} files discovered in {root}")

        stats = {"files_scanned": 0, "chunks_indexed": 0, "chunks_skipped": 0}

        try:
            for file_path in files:
                source_name = file_path.parent.name
                try:
                    indexed, skipped = self._ingest_file(file_path, source_name, namespace)
                    stats["files_scanned"] += 1
                    stats["chunks_indexed"] += indexed
                    stats["chunks_skipped"] += skipped
                except Exception as exc:
                    logger.error(f"Failed to ingest {file_path}: {exc}", exc_info=True)
        finally:
            parsing_cfg["image_output_dir"] = original_img_dir

        if self._run_community_detection and self._community_detector:
            logger.info("Running community detection over full entity graph...")
            try:
                community_stats = self._community_detector.run()
                stats["communities_built"] = community_stats.get("communities_written", 0)
                logger.info("Community detection complete", extra=community_stats)
            except Exception as exc:
                logger.error("Community detection failed (non-fatal): %s", exc, exc_info=True)

        logger.info("Ingestion complete", extra=stats)
        return stats

    def _ingest_file(
        self,
        file_path: Path,
        source_name: str,
        namespace: str | None,
    ) -> tuple[int, int]:
        """Parse, clean, chunk, embed, and upsert one file."""

        # Delta check
        if self._versioning_cfg["delta_ingestion"]:
            file_hash = hash_file(file_path)
            if self._processed_hashes.get(str(file_path)) == file_hash:
                logger.debug(f"Delta skip (unchanged): {file_path.name}")
                return 0, 0
            self._processed_hashes[str(file_path)] = file_hash

        logger.info(f"Ingesting: {file_path.name} [{source_name}]")

        # Parse the file into raw chunks
        raw_chunks: list[ParsedChunk] = self._parser.parse_file(file_path, source_name)
        if not raw_chunks:
            logger.warning(f"No content extracted from {file_path.name}")
            return 0, 0

        # Stamp with ingestion metadata
        ingestion_ts = datetime.now(timezone.utc).isoformat()
        doc_version = extract_version(file_path)
        for chunk in raw_chunks:
            chunk.ingestion_ts = ingestion_ts
            chunk.doc_version = doc_version

        # PII Redaction
        for chunk in raw_chunks:
            original_text = chunk.text
            chunk.text = self._pii_guard.redact(chunk.text, context="ingestion")
            chunk.metadata["sensitivity"] = "redacted" if chunk.text != original_text else "public"

        # Fingerprint
        for chunk in raw_chunks:
            chunk.chunk_id = chunk.compute_fingerprint()

        # Consolidate
        consolidated = self._consolidator.consolidate(raw_chunks)
        if not consolidated:
            logger.warning(f"Consolidation produced no chunks for {file_path.name}")
            return 0, 0

        # Deduplicate
        unique_chunks = self._deduplicator.filter(consolidated)
        skipped = len(consolidated) - len(unique_chunks)

        # Chunk, Embed, and Upsert
        chunk_nodes: list[ChunkNode] = self._chunker.chunk(unique_chunks)

        indexed = 0
        pairs = self._embedder.embed_nodes(chunk_nodes)

        for node, vector in pairs:
            chunk = node.chunk
            chunk.metadata["parent_id"] = node.parent_id
            chunk.metadata["hierarchy_level"] = node.level
            chunk.metadata["namespace"] = namespace or "default"

            # All nodes (doc/section/paragraph) go into Qdrant so we can
            # fetch parents by ID for context expansion. Only paragraphs
            # are searchable — ES and ANN search both filter on
            # hierarchy_level.
            self._dense_store.upsert(chunk, vector, namespace=namespace)

            if node.level == "paragraph":
                self._sparse_index.index_chunk(chunk)
                indexed += 1

        if self._graph_enabled and self._graph_handler:
            self._graph_handler.process([node.chunk for node, _ in pairs], file_path.name)

        logger.info(
            f"File ingested: {file_path.name} → " f"{indexed} chunks indexed, {skipped} skipped"
        )
        return indexed, skipped
