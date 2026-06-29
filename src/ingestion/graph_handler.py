from __future__ import annotations

import asyncio

from src.indexing.embedder import QueryEmbedder
from src.ingestion.parser import ParsedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphIngestionHandler:
    """Handles entity/relationship extraction and entity embedding for one file."""

    def __init__(self, graph_extractor: object, graph_store: object) -> None:
        self._extractor = graph_extractor
        self._store = graph_store

    def process(self, chunks: list[ParsedChunk], file_name: str) -> None:
        """Extract entities and relationships from text chunks and persist to Neo4j."""
        try:
            text_chunks = [c for c in chunks if c.modality == "text" and c.text]
            if not text_chunks:
                return

            extraction_results = asyncio.run(self._extractor.batch_extract(text_chunks))

            entities_written = 0
            relationships_written = 0
            for result in extraction_results:
                self._store.upsert_extraction(result)
                entities_written += len(result.entities)
                relationships_written += len(result.relationships)

            self._embed_new_entities()

            logger.info(
                "GraphRAG extraction complete for %s: "
                "%d entities, %d relationships written to Neo4j",
                file_name,
                entities_written,
                relationships_written,
            )
        except Exception as exc:
            logger.error(
                "GraphRAG extraction failed for %s (non-fatal): %s",
                file_name,
                exc,
                exc_info=True,
            )

    def _embed_new_entities(self) -> None:
        """Embed any entity nodes in Neo4j that are missing an embedding vector."""
        try:
            with self._store._session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.embedding IS NULL
                    RETURN e.node_id AS node_id, e.name AS name,
                           e.description AS description
                    LIMIT 500
                    """
                )
                rows = [dict(r) for r in result]

            if not rows:
                return

            texts = [
                f"{r['name']}: {r['description']}" if r.get("description") else r["name"]
                for r in rows
            ]

            emb = QueryEmbedder()
            pairs: list[tuple[str, list]] = []
            for row, text in zip(rows, texts, strict=False):
                try:
                    vec = emb.embed_query(text)
                    pairs.append((row["node_id"], vec.tolist()))
                except Exception as exc:
                    logger.debug(
                        "Embedding failed for entity %s: %s",
                        row["node_id"][:12],
                        exc,
                    )

            if pairs:
                self._store.batch_update_entity_embeddings(pairs)
                logger.debug("Embedded %d new entity nodes", len(pairs))

        except Exception as exc:
            logger.warning("Entity embedding step failed (non-fatal): %s", exc)
