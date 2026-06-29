from __future__ import annotations

import json

import numpy as np

from agents import RunContextWrapper, function_tool
from src.agents.context import RAGRunContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _chunk_key(item) -> str:
    """Stable dedup key for a context item."""
    cid = getattr(item.chunk, "chunk_id", None)
    return cid if cid else item.chunk.text[:80]


@function_tool
async def graph_query(
    ctx: RunContextWrapper[RAGRunContext],
    mode: str = "hybrid",
) -> str:
    """
    Retrieve context from the knowledge graph (GraphRAG).
    Requires retrieve_context to have run first (needs _query_vector).
    mode: "local" | "global" | "hybrid" (default).
    Merges graph results into ctx.context.context_items.
    """
    pq = ctx.context.processed_query
    if pq is None:
        return json.dumps(
            {
                "error": "processed_query not set",
                "chunks_retrieved": 0,
            }
        )

    query_vector: np.ndarray | None = ctx.context._query_vector
    if query_vector is None:
        return json.dumps(
            {
                "error": "_query_vector not set",
                "chunks_retrieved": 0,
            }
        )

    container = ctx.context.container
    if container is None:
        return json.dumps({"error": "AppContainer not attached to context", "chunks_retrieved": 0})

    gr = container.graph_retriever
    if gr is None:
        return json.dumps(
            {
                "error": "GraphRAG is disabled or unavailable",
                "chunks_retrieved": 0,
            }
        )

    namespace = ctx.context.namespace or "default"

    try:
        graph_items = gr.retrieve(
            pq=pq,
            query_vector=query_vector,
            namespace=namespace,
            mode=mode,
        )
    except Exception as exc:
        logger.error("graph_query tool failed: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc), "chunks_retrieved": 0})

    # Merge: deduplicate against existing vector results
    existing = ctx.context.context_items or []
    existing_keys = {_chunk_key(item) for item in existing}

    new_items = [item for item in graph_items if _chunk_key(item) not in existing_keys]
    ctx.context.context_items = existing + new_items

    sources = list(
        {item.chunk.source_name for item in ctx.context.context_items if item.chunk.source_name}
    )

    logger.info(
        "graph_query tool (%s): %d new graph items merged, total context=%d",
        mode,
        len(new_items),
        len(ctx.context.context_items),
    )

    return json.dumps(
        {
            "chunks_retrieved": len(new_items),
            "sources": sources,
            "retrieval_method": f"graph_{mode}",
            "cache_hit": False,
            "graph_items": len(graph_items),
            "vector_items": len(existing),
        }
    )
