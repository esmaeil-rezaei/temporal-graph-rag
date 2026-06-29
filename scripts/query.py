from __future__ import annotations

import argparse
import asyncio
import sys

from src.core.container import init_container
from src.utils.logger import (
    AMBER as _AMBER,
)
from src.utils.logger import (
    BOLD as _BOLD,
)
from src.utils.logger import (
    CYAN as _CYAN,
)
from src.utils.logger import (
    DIM as _DIM,
)
from src.utils.logger import (
    GREEN as _GREEN,
)
from src.utils.logger import (
    RED as _RED,
)
from src.utils.logger import (
    RESET as _RESET,
)

_HR = f"{_DIM}" + "─" * 60 + f"{_RESET}"


def _print_result(result, question: str) -> None:
    """Render a GenerationResult to the terminal."""
    print()
    print(f"{_BOLD}Query{_RESET}   {question}")
    print(_HR)
    print()
    print(result.answer)
    print()

    if result.citations:
        print(f"{_DIM}Sources ({len(result.citations)}){_RESET}")
        for cite in result.citations:
            src = cite.get("source_name") or cite.get("source_file") or "unknown"
            ts = (cite.get("ingestion_ts") or "")[:10]
            cid = (cite.get("chunk_id") or "")[:8]
            date = f" {_DIM}·{_RESET} {ts}" if ts else ""
            print(f"  {_CYAN}[{cid}…]{_RESET} {src}{date}")
        print()

    if result.faithfulness_score is not None:
        score = result.faithfulness_score
        colour = _GREEN if score >= 0.7 else (_AMBER if score >= 0.4 else _RED)
        bar_filled = int(score * 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        print(f"{_DIM}Faithfulness{_RESET}  {colour}{bar}{_RESET}  {score:.0%}")

    if result.has_conflict and result.conflict_resolution:
        print(f"\n{_AMBER}⚡ Conflict{_RESET}  {result.conflict_resolution}")

    print()


async def _run_query(
    question: str,
    history: list,
    namespace: str | None,
) -> object:
    """Initialise the container and run a single query."""
    container = init_container()
    try:
        return await container.orchestrator.run(
            raw_query=question,
            conversation_history=history,
            namespace=namespace,
        )
    finally:
        container.shutdown()


def run_single_query(question: str, namespace: str | None) -> None:
    result = asyncio.run(_run_query(question, [], namespace))
    _print_result(result, question)


def interactive_repl(namespace: str | None) -> None:
    """Interactive REPL — maintains conversation history across turns."""
    ns_label = f" {_DIM}[{namespace}]{_RESET}" if namespace else ""
    print(
        f"\n{_BOLD}RAG Interactive Mode{_RESET}{ns_label}  "
        f"{_DIM}(type 'exit' or press Ctrl-C to quit){_RESET}\n"
    )

    container = init_container()
    history: list = []

    try:
        while True:
            try:
                question = input(f"{_BOLD}You:{_RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{_DIM}Session ended.{_RESET}")
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit", "q", ":q"}:
                print(f"{_DIM}Goodbye.{_RESET}")
                break

            try:
                result = asyncio.run(
                    container.orchestrator.run(
                        raw_query=question,
                        conversation_history=history,
                        namespace=namespace,
                    )
                )
            except Exception as exc:
                print(f"{_RED}Error:{_RESET} {exc}\n", file=sys.stderr)
                continue

            _print_result(result, question)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result.answer})
    finally:
        container.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the RAG knowledge assistant from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.query 'What are the main findings?'\n"
            "  python -m scripts.query 'Summarise biomarkers' --namespace Radiology\n"
            "  python -m scripts.query --interactive --namespace Alzheimer\n"
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to answer (omit to enter interactive mode)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start an interactive conversation REPL",
    )
    parser.add_argument(
        "--namespace",
        "-n",
        metavar="NS",
        default=None,
        help="Restrict retrieval to a specific namespace (e.g. Radiology)",
    )

    args = parser.parse_args()

    if args.interactive or not args.question:
        interactive_repl(namespace=args.namespace)
    else:
        run_single_query(args.question, namespace=args.namespace)


if __name__ == "__main__":
    main()
