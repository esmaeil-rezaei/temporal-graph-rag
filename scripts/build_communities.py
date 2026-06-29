from __future__ import annotations

import argparse
import sys

from src.config.settings import get_config
from src.utils.logger import get_logger

try:
    from src.graphrag.community import CommunityDetector
    from src.graphrag.neo4j_store import Neo4jGraphStore
except ImportError as _graphrag_import_err:
    Neo4jGraphStore = None  # type: ignore
    CommunityDetector = None  # type: ignore
    _graphrag_import_err_msg = str(_graphrag_import_err)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build knowledge-graph communities from ingested entities"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.parse_args()

    cfg = get_config()
    gr_cfg = cfg.get("graphrag", {})

    if not gr_cfg.get("enabled", False):
        logger.error("GraphRAG is disabled. Set graphrag.enabled: true in config.yaml and re-run.")
        sys.exit(1)

    if Neo4jGraphStore is None:
        logger.error(
            "GraphRAG dependencies not installed: %s\n"
            "Run: pip install neo4j python-louvain cdlib",
            _graphrag_import_err_msg,
        )
        sys.exit(1)

    logger.info("Starting community detection pass...")

    try:
        with Neo4jGraphStore() as graph_store:
            detector = CommunityDetector(graph_store)
            stats = detector.run()

        logger.info(
            "Community detection complete: "
            "%d communities built across %d levels, "
            "%d entities assigned",
            stats["total_communities"],
            stats["levels_run"],
            stats["entities_assigned"],
        )
    except Exception as exc:
        logger.error("Community detection failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
