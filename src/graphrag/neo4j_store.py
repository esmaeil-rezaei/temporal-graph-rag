from __future__ import annotations

import functools
import time
from typing import Any

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable, TransientError

from src.config.settings import get_config, get_secrets
from src.graphrag.schema import (
    CommunityNode,
    EntityNode,
    EntityType,
    ExtractionResult,
    RelationshipEdge,
    RelationshipType,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ENTITY_VECTOR_INDEX = "entity_embeddings"
_DEFAULT_EMBEDDING_DIM = 1024
_MAX_RETRIES = 3


def _with_retry(fn):
    """
    Wrap a Neo4j write function with simple exponential-backoff retry
    for TransientError (deadlocks, leader elections, etc.).
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except (TransientError, ServiceUnavailable) as exc:
                last_exc = exc
                wait = 2**attempt
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %ds",
                    fn.__name__,
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"{fn.__name__} failed after {_MAX_RETRIES} retries: {last_exc}")

    return wrapper


class Neo4jGraphStore:
    """Manages all knowledge-graph persistence in Neo4j."""

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        gr_cfg: dict[str, Any] = cfg.get("graphrag", {})
        neo4j_cfg: dict[str, Any] = gr_cfg.get("neo4j", {})

        uri = neo4j_cfg.get("uri", getattr(sec, "neo4j_uri", "neo4j://localhost:7687"))
        username = neo4j_cfg.get("username", getattr(sec, "neo4j_username", "neo4j"))
        password = neo4j_cfg.get("password", getattr(sec, "neo4j_password", "password"))
        self._database: str = neo4j_cfg.get("database", getattr(sec, "neo4j_database", "neo4j"))
        pool_size: int = neo4j_cfg.get("max_connection_pool_size", 50)
        self._embedding_dim: int = cfg.embeddings.get(
            "embedding_dimensions", _DEFAULT_EMBEDDING_DIM
        )

        self._driver: Driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_pool_size=pool_size,
        )
        logger.info("Neo4j driver initialised: %s (db=%s)", uri, self._database)

        self._ensure_constraints_and_indexes()

    def _ensure_constraints_and_indexes(self) -> None:
        """Create uniqueness constraints and vector index at startup (idempotent)."""
        with self._session() as session:
            session.run(
                "CREATE CONSTRAINT entity_node_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.node_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT community_id IF NOT EXISTS "
                "FOR (c:Community) REQUIRE c.community_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE"
            )
            session.run("CREATE INDEX entity_name IF NOT EXISTS " "FOR (e:Entity) ON (e.name)")
            session.run(
                "CREATE INDEX entity_type_idx IF NOT EXISTS " "FOR (e:Entity) ON (e.entity_type)"
            )

        self._ensure_vector_index()
        logger.info("Neo4j constraints and indexes ensured")

    def _ensure_vector_index(self) -> None:
        """Create the native vector index on Entity.embedding if absent."""
        cypher = (
            f"CREATE VECTOR INDEX {_ENTITY_VECTOR_INDEX} IF NOT EXISTS "
            f"FOR (e:Entity) ON (e.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {self._embedding_dim}, "
            f"`vector.similarity_function`: 'cosine'}}}}"
        )
        try:
            with self._session() as session:
                session.run(cypher)
            logger.info(
                "Vector index '%s' ensured (%d dims, cosine)",
                _ENTITY_VECTOR_INDEX,
                self._embedding_dim,
            )
        except Exception as exc:
            logger.warning(
                "Could not create vector index (Neo4j < 5.11?): %s — "
                "entity vector search will be unavailable",
                exc,
            )

    @_with_retry
    def upsert_extraction(self, result: ExtractionResult) -> None:
        """Persist all entities and relationships from one ExtractionResult in a single transaction."""
        with self._session() as session:
            with session.begin_transaction() as tx:
                for entity in result.entities:
                    self._merge_entity(tx, entity)

                for edge in result.relationships:
                    self._merge_relationship(tx, edge)

                if result.chunk_id:
                    self._merge_chunk_node(tx, result.chunk_id)

                    for entity in result.entities:
                        self._merge_mentions(tx, result.chunk_id, entity.node_id)

                tx.commit()

        logger.debug(
            "Upserted extraction for chunk %s: %d entities, %d relationships",
            result.chunk_id[:12] if result.chunk_id else "?",
            len(result.entities),
            len(result.relationships),
        )

    @_with_retry
    def upsert_community(self, community: CommunityNode) -> None:
        """
        Store a community node and link its member entities to it.
        """
        with self._session() as session:
            with session.begin_transaction() as tx:
                tx.run(
                    """
                    MERGE (c:Community {community_id: $community_id})
                    SET c.level       = $level,
                        c.title       = $title,
                        c.summary     = $summary,
                        c.embedding   = $embedding,
                        c.created_at  = $created_at,
                        c.member_count = $member_count
                    """,
                    community_id=community.community_id,
                    level=community.level,
                    title=community.title,
                    summary=community.summary,
                    embedding=community.embedding,
                    created_at=community.created_at,
                    member_count=len(community.member_ids),
                )
                for member_id in community.member_ids:
                    tx.run(
                        """
                        MATCH (e:Entity {node_id: $node_id})
                        MATCH (c:Community {community_id: $community_id})
                        MERGE (e)-[:BELONGS_TO_COMMUNITY]->(c)
                        """,
                        node_id=member_id,
                        community_id=community.community_id,
                    )
                tx.commit()

    @_with_retry
    def update_entity_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store or update the vector embedding on an Entity node."""
        with self._session() as session:
            session.run(
                "MATCH (e:Entity {node_id: $node_id}) SET e.embedding = $embedding",
                node_id=node_id,
                embedding=embedding,
            )

    @_with_retry
    def batch_update_entity_embeddings(self, pairs: list[tuple[str, list[float]]]) -> None:
        """Bulk update entity embeddings in a single transaction."""
        with self._session() as session:
            with session.begin_transaction() as tx:
                for node_id, embedding in pairs:
                    tx.run(
                        "MATCH (e:Entity {node_id: $node_id}) " "SET e.embedding = $embedding",
                        node_id=node_id,
                        embedding=embedding,
                    )
                tx.commit()

    def get_entity_by_name(
        self, name: str, entity_type: EntityType | None = None
    ) -> dict[str, Any] | None:
        """Fetch a single entity node by name (and optional type)."""
        cypher = "MATCH (e:Entity {name: $name})"
        params: dict[str, Any] = {"name": name}
        if entity_type:
            cypher += " WHERE e.entity_type = $entity_type"
            params["entity_type"] = entity_type.value
        cypher += " RETURN e LIMIT 1"
        with self._session() as session:
            record = session.run(cypher, **params).single()
            return dict(record["e"]) if record else None

    def get_entity_neighbors(
        self,
        node_id: str,
        relation_types: list[RelationshipType] | None = None,
        max_hops: int = 1,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Return neighbors of an entity up to ``max_hops`` hops away.
        Optionally filter by relationship type.

        Returns a list of dicts with keys: entity, relationship, path_length.
        """
        if relation_types:
            rel_filter = "|".join(r.value for r in relation_types)
            rel_pattern = f"-[r:{rel_filter}*1..{max_hops}]->"
        else:
            rel_pattern = f"-[r*1..{max_hops}]->"

        cypher = f"""
            MATCH (start:Entity {{node_id: $node_id}})
            {rel_pattern}(neighbor:Entity)
            RETURN DISTINCT neighbor, length(relationships(r)) AS path_length
            ORDER BY path_length, neighbor.name
            LIMIT $limit
        """
        with self._session() as session:
            result = session.run(cypher, node_id=node_id, limit=limit)
            return [
                {"entity": dict(row["neighbor"]), "path_length": row["path_length"]}
                for row in result
            ]

    def get_entity_subgraph(
        self,
        node_ids: list[str],
        max_hops: int = 2,
    ) -> dict[str, Any]:
        """Return the induced subgraph for a set of entity node_ids."""
        cypher = f"""
            MATCH (e:Entity)
            WHERE e.node_id IN $node_ids
            OPTIONAL MATCH (e)-[r*1..{max_hops}]-(neighbor:Entity)
            RETURN collect(DISTINCT e) + collect(DISTINCT neighbor) AS nodes,
                   collect(DISTINCT r) AS edges
        """
        with self._session() as session:
            record = session.run(cypher, node_ids=node_ids).single()
            if not record:
                return {"nodes": [], "edges": []}
            return {
                "nodes": [dict(n) for n in record["nodes"] if n],
                "edges": [dict(r) for r in record["edges"] if r],
            }

    def vector_search_entities(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        entity_type: EntityType | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search over entity embeddings using the Neo4j vector index; returns {entity, score} list."""
        filter_clause = (
            f"WHERE candidate.entity_type = '{entity_type.value}'" if entity_type else ""
        )
        cypher = f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node AS candidate, score
            {filter_clause}
            RETURN candidate, score
            ORDER BY score DESC
        """
        with self._session() as session:
            result = session.run(
                cypher,
                index_name=_ENTITY_VECTOR_INDEX,
                top_k=top_k,
                embedding=query_embedding,
            )
            return [{"entity": dict(row["candidate"]), "score": row["score"]} for row in result]

    def get_community_summary(self, community_id: str) -> str | None:
        """Return the LLM-generated summary for a community node."""
        with self._session() as session:
            record = session.run(
                "MATCH (c:Community {community_id: $id}) RETURN c.summary AS summary",
                id=community_id,
            ).single()
            return record["summary"] if record else None

    def get_chunks_for_entity(self, node_id: str) -> list[str]:
        """Return the chunk_ids of all chunks that mention this entity."""
        with self._session() as session:
            result = session.run(
                """
                MATCH (ch:Chunk)-[:MENTIONS]->(e:Entity {node_id: $node_id})
                RETURN ch.chunk_id AS chunk_id
                """,
                node_id=node_id,
            )
            return [row["chunk_id"] for row in result if row["chunk_id"]]

    def get_entity_communities(self, node_id: str) -> list[dict[str, Any]]:
        """Return community nodes an entity belongs to (all levels)."""
        with self._session() as session:
            result = session.run(
                """
                MATCH (e:Entity {node_id: $node_id})-[:BELONGS_TO_COMMUNITY]->(c:Community)
                RETURN c ORDER BY c.level
                """,
                node_id=node_id,
            )
            return [dict(row["c"]) for row in result]

    def get_all_entities(self, limit: int = 10_000) -> list[dict[str, Any]]:
        """Bulk fetch entity nodes for community detection."""
        with self._session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                RETURN e.node_id AS node_id, e.name AS name,
                       e.entity_type AS entity_type
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(row) for row in result]

    def get_all_relationships(self, limit: int = 100_000) -> list[dict[str, Any]]:
        """Bulk fetch edges for community detection."""
        with self._session() as session:
            result = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                RETURN a.node_id AS source_id, b.node_id AS target_id,
                       type(r) AS relation_type,
                       coalesce(r.weight, 1.0) AS weight
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(row) for row in result]

    def close(self) -> None:
        """Close the Neo4j driver and release all pooled connections."""
        self._driver.close()
        logger.info("Neo4j driver closed")

    def __enter__(self) -> Neo4jGraphStore:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _session(self) -> Session:
        return self._driver.session(database=self._database)

    @staticmethod
    def _merge_entity(tx, entity: EntityNode) -> None:
        tx.run(
            """
            MERGE (e:Entity {node_id: $node_id})
            ON CREATE SET
                e.created_at    = $created_at,
                e.source_chunks = $source_chunks
            ON MATCH SET
                e.source_chunks = e.source_chunks + [x IN $source_chunks WHERE NOT x IN e.source_chunks]
            SET e.name          = $name,
                e.entity_type   = $entity_type,
                e.description   = $description,
                e.confidence    = $confidence,
                e.updated_at    = $updated_at,
                e.properties    = $properties
            """,
            node_id=entity.node_id,
            name=entity.name,
            entity_type=(
                entity.entity_type.value
                if hasattr(entity.entity_type, "value")
                else str(entity.entity_type)
            ),
            description=entity.description,
            confidence=entity.confidence,
            updated_at=entity.updated_at,
            created_at=entity.created_at,
            source_chunks=entity.source_chunks,
            properties=str(entity.properties),
        )
        et_value = (
            entity.entity_type.value
            if hasattr(entity.entity_type, "value")
            else str(entity.entity_type)
        )
        tx.run(
            f"MATCH (e:Entity {{node_id: $node_id}}) SET e:{et_value}",
            node_id=entity.node_id,
        )

    @staticmethod
    def _merge_relationship(tx, edge: RelationshipEdge) -> None:
        rel_type = (
            edge.relation_type.value
            if hasattr(edge.relation_type, "value")
            else str(edge.relation_type)
        )
        tx.run(
            f"""
            MATCH (a:Entity {{node_id: $source_id}})
            MATCH (b:Entity {{node_id: $target_id}})
            MERGE (a)-[r:{rel_type} {{edge_id: $edge_id}}]->(b)
            ON CREATE SET r.created_at = $created_at
            SET r.description   = $description,
                r.weight        = coalesce(r.weight, 0.0) + $weight,
                r.updated_at    = $updated_at,
                r.source_chunks = coalesce(r.source_chunks, []) +
                    [x IN $source_chunks WHERE NOT x IN coalesce(r.source_chunks, [])]
            """,
            source_id=edge.source_id,
            target_id=edge.target_id,
            edge_id=edge.edge_id,
            description=edge.description,
            weight=edge.weight,
            created_at=edge.created_at,
            updated_at=edge.updated_at,
            source_chunks=edge.source_chunks,
        )

    @staticmethod
    def _merge_chunk_node(tx, chunk_id: str) -> None:
        tx.run(
            "MERGE (ch:Chunk {chunk_id: $chunk_id})",
            chunk_id=chunk_id,
        )

    @staticmethod
    def _merge_mentions(tx, chunk_id: str, entity_node_id: str) -> None:
        tx.run(
            """
            MATCH (ch:Chunk {chunk_id: $chunk_id})
            MATCH (e:Entity {node_id: $node_id})
            MERGE (ch)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id,
            node_id=entity_node_id,
        )
