#!/usr/bin/env python3
"""Developer helper for suggesting candidate chunk_ids to populate relevant_chunk_ids in tests/golden_queries.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.core.container import init_container
from src.utils.logger import get_logger

logger = get_logger("suggest_chunk_ids")


def load_golden_queries(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        logger.error("Golden queries file not found: %s", path)
        sys.exit(1)
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def count_keyword_hits(text: str, keywords: list[str]) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Suggest candidate chunk_ids for tests/golden_queries.json (read-only).",
    )
    p.add_argument(
        "--golden",
        default="tests/golden_queries.json",
        metavar="FILE",
        help="Path to golden queries JSON (default: tests/golden_queries.json)",
    )
    p.add_argument("--top-k", type=int, default=10, help="Candidates to retrieve per query")
    p.add_argument(
        "--query-index",
        type=int,
        default=None,
        metavar="N",
        help="Only process the Nth query (0-indexed). Default: all queries.",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write the report to this Markdown file instead of stdout",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    golden = load_golden_queries(args.golden)
    if args.query_index is not None:
        if not (0 <= args.query_index < len(golden)):
            logger.error("--query-index %d out of range (0-%d)", args.query_index, len(golden) - 1)
            sys.exit(1)
        golden = [golden[args.query_index]]

    logger.info("Initialising container (this loads the embedding model)...")
    container = init_container()

    lines: list[str] = []
    lines.append("# Chunk ID Suggestions")
    lines.append("")
    lines.append(
        "Review the candidates below and copy the chunk_ids that are truly "
        "relevant into `relevant_chunk_ids` for each query in "
        f"`{args.golden}`. ★ marks chunks whose text contains at least one "
        "expected answer keyword."
    )
    lines.append("")

    for i, gq in enumerate(golden):
        query = gq["query"]
        namespace = gq.get("namespace", "default")
        expected_keywords = gq.get("expected_answer_keywords", [])

        logger.info("[%d/%d] Searching: %s", i + 1, len(golden), query[:70])

        query_vector = container.embedder.embed_query(query)
        results = container.dense_store.search(
            query_vector=query_vector,
            top_k=args.top_k,
            namespace=namespace,
        )

        lines.append(f"## Query {i}: {query}")
        lines.append("")
        lines.append(f"- namespace: `{namespace}`")
        lines.append(f"- expected_answer_keywords: {expected_keywords}")
        lines.append("")

        if not results:
            lines.append("_No candidates returned._")
            lines.append("")
            continue

        lines.append("| # | chunk_id | score | keyword hits | preview |")
        lines.append("|---|---|---|---|---|")
        for rank, r in enumerate(results, start=1):
            chunk_id = r.chunk.chunk_id or "—"
            hits = count_keyword_hits(r.chunk.text, expected_keywords)
            flag = "★" if hits else ""
            preview = r.chunk.text.replace("\n", " ").strip()[:100]
            lines.append(
                f"| {rank}{flag} | `{chunk_id}` | {r.score:.4f} | "
                f"{', '.join(hits) if hits else '—'} | {preview}… |"
            )
        lines.append("")

    report = "\n".join(lines)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        logger.info("Report written to %s", args.output)
    else:
        print(report)


if __name__ == "__main__":
    main()
