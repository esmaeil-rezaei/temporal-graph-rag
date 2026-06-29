import argparse
import logging
from pathlib import Path

from src.config.settings import get_config
from src.ingestion.pipeline import IngestionPipeline
from src.utils.logger import get_logger, set_correlation_id

logging.getLogger("pdfminer").setLevel(logging.ERROR)

logger = get_logger("scripts.ingest")


def _discover_namespaces() -> list[str]:
    cfg = get_config()
    kb_root = Path(cfg.knowledge_base["root_dir"])
    if not kb_root.exists():
        return []
    return sorted([d.name for d in kb_root.iterdir() if d.is_dir() and not d.name.startswith(".")])


def _abort_namespace(given: str, available: list[str]) -> None:
    kb_root = get_config().knowledge_base["root_dir"]
    avail_str = (
        ", ".join(available) if available else f"(none — create a subfolder inside {kb_root}/)"
    )
    logger.error(
        "Namespace '%s' does not exist. Expected folder: %s/%s/  Available: %s  "
        "Tip: run with --all to ingest every namespace automatically.",
        given,
        kb_root,
        given,
        avail_str,
    )
    raise SystemExit(1)


def _abort_no_namespaces() -> None:
    kb_root = get_config().knowledge_base["root_dir"]
    logger.error(
        "No namespace folders found. Create at least one subfolder inside %s/ and add documents to it.",
        kb_root,
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ingestion pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Ingest a single namespace folder (e.g. --namespace AD)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Auto-discover and ingest every top-level knowledge_base/ folder as its own namespace",
    )
    args = parser.parse_args()

    set_correlation_id()

    available = _discover_namespaces()

    if args.all:
        if not available:
            _abort_no_namespaces()
        logger.info(f"Discovered namespaces: {available}")
        pipeline = IngestionPipeline()
        for ns in available:
            logger.info(f"=== Ingesting namespace: {ns} ===")
            stats = pipeline.run(namespace=ns)
            logger.info(f"Namespace '{ns}' complete: {stats}")

    elif args.namespace:
        if args.namespace not in available:
            _abort_namespace(args.namespace, available)
        logger.info(f"Starting ingestion (namespace={args.namespace})")
        pipeline = IngestionPipeline()
        stats = pipeline.run(namespace=args.namespace)
        logger.info(f"Ingestion complete: {stats}")

    else:
        kb_root = get_config().knowledge_base["root_dir"]
        avail_str = (
            ", ".join(available)
            if available
            else f"(none found — create a subfolder inside {kb_root}/)"
        )
        print(
            f"\nNo namespace specified.\n"
            f"Usage:  python -m scripts.ingest --namespace <name>\n"
            f"\nThe namespace must match a folder inside your knowledge base root: {kb_root}/\n"
            f"Available namespaces: {avail_str}\n"
            f"\nExample: python -m scripts.ingest --namespace {available[0] if available else 'my_project'}\n"
            f"         python -m scripts.ingest --all   (ingest all namespaces at once)\n"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
