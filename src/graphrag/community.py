from __future__ import annotations

import hashlib
from typing import Any

import networkx as nx
import openai

from src.config.settings import get_config, get_secrets
from src.graphrag.neo4j_store import Neo4jGraphStore
from src.graphrag.schema import CommunityNode
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import community as community_louvain  # type: ignore  # python-louvain

    _LOUVAIN_AVAILABLE = True
except ImportError:
    _LOUVAIN_AVAILABLE = False
    logger.warning("python-louvain not installed; will try cdlib Leiden")

try:
    from cdlib import algorithms as cdlib_algos  # type: ignore

    _CDLIB_AVAILABLE = True
except ImportError:
    _CDLIB_AVAILABLE = False

if not _LOUVAIN_AVAILABLE and not _CDLIB_AVAILABLE:
    logger.warning(
        "Neither python-louvain nor cdlib is available. "
        "Community detection will fall back to connected-components (low quality)."
    )

_COMMUNITY_SUMMARY_SYSTEM = (
    "You are a knowledge-graph analyst. "
    "Given a list of entities and their descriptions, write a concise (3-5 sentence) "
    "summary of what this group of entities has in common and what topic or theme they represent. "
    "Return ONLY the summary text — no preamble, no bullet points."
)

_COMMUNITY_SUMMARY_USER = """\
Entities in this community:
{entity_list}

Write a thematic summary of this community.
"""


class CommunityDetector:
    """Detects communities in the entity graph and writes summaries back to Neo4j."""

    def __init__(self, graph_store: Neo4jGraphStore) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._store = graph_store
        self._gr_cfg: dict[str, Any] = cfg.get("graphrag", {})
        self._community_cfg: dict[str, Any] = self._gr_cfg.get("community", {})
        self._min_community_size: int = self._community_cfg.get("min_community_size", 3)
        self._resolution_levels: list[float] = self._community_cfg.get(
            "resolution_levels", [1.0, 0.5]  # level 0 = fine, level 1 = coarse
        )
        self._summary_model: str = self._community_cfg.get("summary_model", "gpt-4o")
        self._max_entities_per_summary: int = self._community_cfg.get(
            "max_entities_per_summary", 30
        )
        self._openai = openai.OpenAI(api_key=sec.openai_api_key)

    def run(self) -> dict[str, int]:
        """Run full community detection and return stats {total_communities, entities_assigned, levels_run}."""
        logger.info("Community detection started")

        nodes = self._store.get_all_entities()
        edges = self._store.get_all_relationships()

        if not nodes:
            logger.warning("No entities found — skipping community detection")
            return {"total_communities": 0, "entities_assigned": 0, "levels_run": 0}

        G = self._build_graph(nodes, edges)
        logger.info("Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

        stats = {"total_communities": 0, "entities_assigned": 0, "levels_run": 0}

        for level, resolution in enumerate(self._resolution_levels):
            communities = self._detect(G, resolution=resolution)
            if not communities:
                continue

            communities = {
                cid: members
                for cid, members in communities.items()
                if len(members) >= self._min_community_size
            }

            logger.info(
                "Level %d (resolution=%.2f): %d communities (≥%d members)",
                level,
                resolution,
                len(communities),
                self._min_community_size,
            )

            for cid_local, member_node_ids in communities.items():
                community_id = self._stable_community_id(level, cid_local)

                member_meta = [n for n in nodes if n["node_id"] in member_node_ids]

                summary, title = None, None
                if level == 0:
                    summary, title = self._generate_summary(member_meta)

                community = CommunityNode(
                    community_id=community_id,
                    level=level,
                    member_ids=list(member_node_ids),
                    summary=summary,
                    title=title,
                )
                self._store.upsert_community(community)
                stats["total_communities"] += 1
                stats["entities_assigned"] += len(member_node_ids)

            stats["levels_run"] += 1

        logger.info("Community detection complete: %s", stats)
        return stats

    @staticmethod
    def _build_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> nx.Graph:
        """Build an undirected weighted NetworkX graph from entity nodes and relationship edges."""
        G: nx.Graph = nx.Graph()
        for node in nodes:
            G.add_node(node["node_id"], **node)
        for edge in edges:
            src = edge.get("source_id")
            tgt = edge.get("target_id")
            w = float(edge.get("weight", 1.0))
            if src and tgt:
                if G.has_edge(src, tgt):
                    G[src][tgt]["weight"] += w
                else:
                    G.add_edge(src, tgt, weight=w)
        return G

    def _detect(self, G: nx.Graph, resolution: float = 1.0) -> dict[str, set]:
        """Detect communities and return {community_label → set(node_ids)}."""
        if _CDLIB_AVAILABLE:
            return self._detect_leiden(G, resolution)
        elif _LOUVAIN_AVAILABLE:
            return self._detect_louvain(G, resolution)
        else:
            return self._detect_components(G)

    @staticmethod
    def _detect_louvain(G: nx.Graph, resolution: float) -> dict[str, set]:
        """Louvain community detection."""
        partition: dict[str, int] = community_louvain.best_partition(
            G, weight="weight", resolution=resolution
        )
        communities: dict[str, set] = {}
        for node_id, comm_id in partition.items():
            communities.setdefault(str(comm_id), set()).add(node_id)
        return communities

    @staticmethod
    def _detect_leiden(G: nx.Graph, resolution: float) -> dict[str, set]:
        """Leiden community detection (more stable than Louvain)."""
        result = cdlib_algos.leiden(G, weights="weight")  # type: ignore
        communities: dict[str, set] = {}
        for i, community_list in enumerate(result.communities):
            communities[str(i)] = set(community_list)
        return communities

    @staticmethod
    def _detect_components(G: nx.Graph) -> dict[str, set]:
        """Fallback: use connected components (low quality but always available)."""
        logger.warning("Using connected-components as community detection fallback")
        return {str(i): set(comp) for i, comp in enumerate(nx.connected_components(G))}

    def _generate_summary(self, member_meta: list[dict[str, Any]]) -> tuple[str | None, str | None]:
        """Generate a thematic LLM summary and title for a community; returns (None, None) on failure."""
        if not member_meta:
            return None, None

        # Truncate large communities to keep the prompt manageable
        sample = member_meta[: self._max_entities_per_summary]
        entity_list = "\n".join(
            f"- {m['name']} ({m.get('entity_type', 'UNKNOWN')})" for m in sample
        )

        try:
            response = self._openai.chat.completions.create(
                model=self._summary_model,
                messages=[
                    {"role": "system", "content": _COMMUNITY_SUMMARY_SYSTEM},
                    {
                        "role": "user",
                        "content": _COMMUNITY_SUMMARY_USER.format(entity_list=entity_list),
                    },
                ],
                temperature=0.0,
                max_tokens=300,
            )
            summary = response.choices[0].message.content.strip()

            title = summary.split(".")[0].strip()[:80] if summary else None
            return summary, title

        except Exception as exc:
            logger.warning("Community summary generation failed: %s", exc)
            return None, None

    @staticmethod
    def _stable_community_id(level: int, local_id: str) -> str:
        """Generate a stable deterministic community_id from level + partition id via SHA-256."""
        key = f"community::level{level}::{local_id}"
        return hashlib.sha256(key.encode()).hexdigest()
