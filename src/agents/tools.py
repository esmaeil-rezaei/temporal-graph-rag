from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from agents import RunContextWrapper, function_tool
from src.agents.context import RAGRunContext
from src.config.settings import get_config
from src.operations.ops_middleware import SemanticCache, TraceSpan
from src.query.understanding import ProcessedQuery
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)
_cfg = get_config()


def _run_online_eval_bg(ev, raw_q: str, answer: str, context_texts: list[str]) -> None:
    """Run RAGAS online evaluation in a background thread."""
    try:
        report = ev.evaluate_online(
            query=raw_q,
            answer=answer,
            context_texts=context_texts,
        )
        logger.info(
            "Online evaluation complete",
            extra={
                "overall_score": report.overall_score,
                "scores": report.ragas_scores or report.custom_judge_scores,
            },
        )
    except Exception as exc:
        logger.warning("Online evaluation failed (non-fatal): %s", exc)


def _container(ctx: RunContextWrapper[RAGRunContext]):
    c = ctx.context.container
    if c is None:
        raise RuntimeError("AppContainer not attached to RAGRunContext")
    return c


@function_tool
async def get_conversation_history(
    ctx: RunContextWrapper[RAGRunContext],
) -> str:
    return json.dumps(
        {
            "conversation_history": (
                ctx.context.conversation_history[-5:] if ctx.context.conversation_history else []
            )
        }
    )


@function_tool
async def get_routing_intent(
    ctx: RunContextWrapper[RAGRunContext],
) -> str:
    """
    Returns the pre-computed routing intent for the current query.
    Use this instead of examining conversation history directly.
    Values: 'conversational' | 'followup' | 'retrieval'
    """
    return json.dumps(
        {
            "query_routing_intent": ctx.context.query_routing_intent,
            "has_history": bool(ctx.context.conversation_history),
        }
    )


@function_tool
async def prepare_query(
    ctx: RunContextWrapper[RAGRunContext],
    query: str,
) -> str:
    """Run query understanding: rewrite, decompose, HyDE, entity filters."""
    if not getattr(ctx.context, "correlation_id", None):
        ctx.context.correlation_id = set_correlation_id()

    logger.info(
        "understand_query tool started",
        extra={"query": query, "correlation_id": ctx.context.correlation_id},
    )

    qu = _container(ctx).query_understanding

    if isinstance(ctx.context.processed_query, str) and ctx.context.processed_query:
        effective_query = ctx.context.processed_query
    else:
        effective_query = query or ctx.context.raw_query

    with TraceSpan("understand_query"):
        try:
            pq = qu.process(
                query=effective_query,
                raw_query=ctx.context.raw_query,
                conversation_history=ctx.context.conversation_history,
            )
            ctx.context.processed_query = pq
            ctx.context.record("understand_query_tool", f"{len(pq.sub_questions)} sub-questions")

            sub_questions = [sq if isinstance(sq, str) else sq.text for sq in pq.sub_questions]

            if pq.hypothetical_doc is not None:
                logger.info(
                    "HyDE document generated — dual retrieval will be used",
                    extra={"hyde_preview": pq.hypothetical_doc[:120]},
                )

            return json.dumps(
                {
                    "standalone_query": pq.standalone_query,
                    "sub_questions": sub_questions,
                    "search_queries": [pq.standalone_query] + sub_questions,
                    "requires_hyde": pq.hypothetical_doc is not None,
                    "metadata_filters": pq.metadata_filters,
                }
            )
        except Exception as exc:
            logger.error("understand_query tool failed: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc), "standalone_query": query})


@function_tool
async def retrieve_context(
    ctx: RunContextWrapper[RAGRunContext],
) -> str:
    """Embed the query, check semantic cache, run hybrid multi-query retrieval."""
    pq = ctx.context.processed_query
    if pq is None:
        return json.dumps({"error": "processed_query not set — run prepare_query first"})

    container = _container(ctx)
    embedder = container.embedder
    retriever = container.retriever
    evaluator = container.evaluator

    namespace = ctx.context.namespace or "default"

    with TraceSpan("query_embedding"):
        try:
            query_vector: np.ndarray = embedder.embed_query(pq.final_query(), language=pq.language)

            hyde_vector: np.ndarray | None = None
            if pq.hypothetical_doc is not None:
                hyde_vector = embedder.embed_query(pq.hypothetical_doc, language=pq.language)
                logger.info(
                    "HyDE vector generated — dual retrieval will be used",
                    extra={"hyde_preview": pq.hypothetical_doc[:120]},
                )

            sub_vectors = []
            for sq in pq.sub_questions:
                try:
                    sv = embedder.embed_query(sq, language=pq.language)
                    sub_vectors.append((sq, sv))
                except Exception as exc:
                    logger.warning("Sub-question embedding failed for '%s': %s", sq[:60], exc)

            if sub_vectors:
                logger.info(
                    "Sub-question vectors generated: %d / %d",
                    len(sub_vectors),
                    len(pq.sub_questions),
                )

        except Exception as exc:
            logger.error("Embedding failed: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc), "chunks_retrieved": 0})

    with TraceSpan("cache_lookup"):
        cache = SemanticCache()
        cached = cache.get(
            query_vector,
            query_routing_intent=ctx.context.query_routing_intent,
            namespace=namespace,
        )

    if cached is not None:
        logger.info("Cache hit — skipping retrieval and generation")
        ctx.context.generation_result = cached
        ctx.context.record("retrieve_context_tool", "cache_hit")
        return json.dumps(
            {
                "chunks_retrieved": 0,
                "sources": [],
                "retrieval_method": "cache",
                "cache_hit": True,
            }
        )

    try:
        evaluator.update_reference_distribution(query_vector)
        ctx.context._evaluator = evaluator
    except Exception as exc:
        logger.warning("Evaluator reference update failed (non-fatal): %s", exc)
        ctx.context._evaluator = None

    with TraceSpan("retrieval", {"namespace": namespace}):
        try:
            if sub_vectors:
                context_items = retriever.retrieve_multi_query(
                    pq=pq,
                    query_vector=query_vector,
                    sub_vectors=sub_vectors,
                    hyde_vector=hyde_vector,  # may be None — retriever handles it
                    namespace=namespace,
                )
                method = "multi_query"
            elif hyde_vector is not None:
                context_items = retriever.retrieve_dual(
                    pq=pq,
                    query_vector=query_vector,
                    hyde_vector=hyde_vector,
                    namespace=namespace,
                )
                method = "hyde_dual"
            else:
                context_items = retriever.retrieve(
                    pq=pq,
                    query_vector=query_vector,
                    namespace=namespace,
                )
                method = "dense"

            ctx.context.context_items = context_items
            ctx.context._query_vector = query_vector
            ctx.context.record("retrieve_context_tool", f"{len(context_items)} chunks via {method}")

            sources = list(
                {item.chunk.source_name for item in context_items if item.chunk.source_name}
            )

            return json.dumps(
                {
                    "chunks_retrieved": len(context_items),
                    "sources": sources,
                    "retrieval_method": method,
                    "cache_hit": False,
                }
            )

        except Exception as exc:
            logger.error("retrieve_context tool failed: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc), "chunks_retrieved": 0})


@function_tool
async def generate_from_history(
    ctx: RunContextWrapper[RAGRunContext],
) -> str:
    """
    Answer a follow-up question using conversation history only.
    Returns needs_retrieval=true when history is absent or clearly insufficient,
    so the caller can hand off to RetrievalAgent instead.
    """
    history = ctx.context.conversation_history

    if not history:
        return json.dumps({"answer": "", "needs_retrieval": True})

    condensed = ctx.context.processed_query
    question = condensed if isinstance(condensed, str) and condensed else ctx.context.raw_query

    container = _container(ctx)
    openai_client = container.generator._openai
    gen_cfg = get_config().generation

    history_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in history[-10:])

    system_prompt = (
        "You are a precise question-answering assistant.\n"
        "The conversation history below is your ONLY knowledge source.\n"
        "Your job:\n"
        "1. Find the answer to the user's question inside the history.\n"
        "2. Return that answer directly and concisely — no preamble, "
        "no 'based on the conversation', no hedging.\n"
        "3. If the answer is genuinely not present in the history, "
        "respond with exactly the token: NEEDS_RETRIEVAL"
    )
    user_prompt = f"CONVERSATION HISTORY:\n{history_text}\n\n" f"QUESTION: {question}"

    try:
        response = openai_client.chat.completions.create(
            model=gen_cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=gen_cfg.get("max_tokens", 1024),
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("generate_from_history failed: %s", exc, exc_info=True)
        return json.dumps({"answer": "", "needs_retrieval": True})

    if answer == "NEEDS_RETRIEVAL" or not answer:
        return json.dumps({"answer": "", "needs_retrieval": True})

    from src.generation.generator import GenerationResult

    ctx.context.generation_result = GenerationResult(
        answer=answer,
        model_used="FollowUpAgent",
    )
    return json.dumps({"answer": answer, "needs_retrieval": False})


@function_tool
async def generate_answer(
    ctx: RunContextWrapper[RAGRunContext],
) -> str:
    """Generate a grounded answer from retrieved context."""
    pq = ctx.context.processed_query
    context_items = ctx.context.context_items

    if not context_items:
        unanswered_dir = Path(_cfg.log.get("unknown_query_dir", "logs/unknown"))
        unanswered_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": ctx.context.raw_query,
            "reason": "No relevant context found",
            "conversation_history": ctx.context.conversation_history,
        }
        fname = (
            unanswered_dir
            / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        )
        try:
            fname.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception as exc:
            logger.warning("Failed to write unanswered query log: %s", exc)

        return json.dumps(
            {
                "answer": "I could not find relevant information to answer your question.",
                "citations": [],
                "faithfulness_score": None,
                "has_conflict": False,
            }
        )

    container = _container(ctx)
    generator = container.generator
    pii_guard = container.pii_guard

    with TraceSpan("generation"):
        try:
            query_text = (
                pq.original_query if isinstance(pq, ProcessedQuery) else ctx.context.raw_query
            )
            user_preferences = getattr(ctx.context, "user_preferences", None) or []
            result = generator.generate(
                query=query_text,
                context_items=context_items,
                extra_instructions=user_preferences or None,
            )
            ctx.context.generation_result = result
            ctx.context.record("generate_answer_tool", f"faithfulness={result.faithfulness_score}")
        except Exception as exc:
            logger.error("generate_answer tool failed during generation: %s", exc, exc_info=True)
            return json.dumps(
                {
                    "answer": "An error occurred while generating the answer.",
                    "citations": [],
                    "faithfulness_score": None,
                    "has_conflict": False,
                }
            )

    with TraceSpan("output_pii_scan"):
        try:
            result.answer = pii_guard.redact(result.answer, context="output")
        except Exception as exc:
            logger.warning("Output PII scan failed (non-fatal): %s", exc)

    with TraceSpan("cache_store"):
        query_vector = ctx.context._query_vector
        if query_vector is not None:
            try:
                cache = SemanticCache()
                cache.put(query_vector, result, namespace=ctx.context.namespace or None)
            except Exception as exc:
                logger.warning("Cache store failed (non-fatal): %s", exc)
        else:
            logger.warning("No _query_vector in context — skipping cache store")

    try:
        ev = ctx.context._evaluator or container.evaluator
        context_texts = [item.chunk.text for item in context_items]
        raw_q = pq.original_query if isinstance(pq, ProcessedQuery) else ctx.context.raw_query
        threading.Thread(
            target=_run_online_eval_bg,
            args=(ev, raw_q, result.answer, context_texts),
            daemon=True,
            name="ragas-online-eval",
        ).start()
    except Exception as exc:
        logger.warning("Online evaluation failed to start (non-fatal): %s", exc)

    logger.info(
        "generate_answer tool complete",
        extra={
            "faithfulness": result.faithfulness_score,
            "citations": len(result.citations),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "has_conflict": result.has_conflict,
        },
    )

    return json.dumps(
        {
            "answer": result.answer,
            "citations": result.citations,
            "faithfulness_score": result.faithfulness_score,
            "has_conflict": result.has_conflict,
        }
    )
