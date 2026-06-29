from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import openai
import spacy
from langdetect import detect as _langdetect

from src.config.settings import get_config, get_secrets
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedQuery:
    """
    The output of the query understanding stage.
    Carries the rewritten query, sub-questions, and metadata filters.
    """

    original_query: str
    expanded_query: str = ""
    standalone_query: str = ""
    sub_questions: list[str] = field(default_factory=list)
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    hypothetical_doc: str | None = None
    language: str | None = None
    query_routing_intent: str = "retrieval"

    def final_query(self) -> str:
        """Return the best query string to use for retrieval."""
        return self.expanded_query or self.standalone_query or self.original_query


class QueryUnderstanding:
    """
    Applies the full query understanding pipeline to a raw user query.
    Each step is enabled/disabled via config.yaml.
    """

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._q_cfg = cfg.query
        self._openai = openai.OpenAI(api_key=sec.openai_api_key)

        self._nlp: spacy.Language | None = None
        if self._q_cfg["entity_recognition"]["enabled"]:
            try:
                self._nlp = spacy.load(self._q_cfg["entity_recognition"]["model"])
            except OSError:
                logger.warning(
                    "spaCy model not found. Run: python -m spacy download en_core_web_trf"
                )

    def process(
        self,
        query: str,
        raw_query: str,
        conversation_history: list[dict[str, str]],
    ) -> ProcessedQuery:
        """Run the full query understanding pipeline on a raw query."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        pq = ProcessedQuery(
            original_query=raw_query,
            standalone_query=query,
        )

        pq.query_routing_intent = self._classify_routing_intent(
            raw_query, ctx_history=conversation_history or []
        )

        working_query = pq.standalone_query

        if self._q_cfg["expansion"]["enabled"]:
            pq.expanded_query, pq.hypothetical_doc = self._expand_query(working_query)
        else:
            pq.expanded_query = working_query

        if self._q_cfg["decomposition"]["enabled"]:
            pq.sub_questions = self._decompose_query(working_query)

        if self._q_cfg["entity_recognition"]["enabled"]:
            pq.metadata_filters = self._extract_filters(working_query)

        try:
            pq.language = _langdetect(query)
        except Exception:
            pq.language = "en"

        logger.info(
            "Query processed",
            extra={
                "original": query,
                "standalone": pq.standalone_query,
                "sub_questions": len(pq.sub_questions),
                "filters": pq.metadata_filters,
            },
        )
        return pq

    def get_routing_intent(self, query: str, history: list[dict[str, str]]) -> str:
        """
        Public wrapper around the rule-based routing classifier.
        Returns 'conversational', 'followup', or 'retrieval'.
        """
        return self._classify_routing_intent(query, ctx_history=history)

    def compress_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """Summarise overflow turns into a [CONVERSATION SUMMARY] system message.
        Controlled by query.conversation.compress_history in config.yaml."""
        conv_cfg = self._q_cfg.get("conversation", {})
        if not conv_cfg.get("compress_history", False):
            return history

        threshold: int = conv_cfg.get("compress_threshold", 14)
        window: int = conv_cfg.get("history_window", 6)
        model: str = conv_cfg.get("compress_model", "gpt-4o-mini")
        max_tok: int = conv_cfg.get("compress_max_tokens", 400)

        if len(history) <= threshold:
            return history

        split = max(0, len(history) - window)
        old_turns = history[:split]
        recent_turns = history[split:]

        if (
            old_turns
            and old_turns[0].get("role") == "system"
            and old_turns[0].get("content", "").startswith("[CONVERSATION SUMMARY]")
        ):
            existing_summary = old_turns[0]["content"]
            turns_to_add = old_turns[1:]
        else:
            existing_summary = None
            turns_to_add = old_turns

        turns_text = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns_to_add)

        if existing_summary:
            user_content = (
                f"Existing summary:\n{existing_summary}\n\n"
                f"New turns to incorporate:\n{turns_text}\n\n"
                "Update the summary to include the new turns. "
                "Preserve all key facts, answers, and user preferences."
            )
        else:
            user_content = (
                f"Conversation turns:\n{turns_text}\n\n"
                "Write a concise factual summary. "
                "Preserve key questions, answers, and user preferences stated. "
                "Do not answer questions — only summarise what was said."
            )

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation compressor. "
                            "Summarise the provided turns into a dense, factual paragraph."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=max_tok,
            )
            summary_text = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("History compression failed (%s); keeping full history.", exc)
            return history

        compressed = [
            {"role": "system", "content": f"[CONVERSATION SUMMARY]\n{summary_text}"}
        ] + list(recent_turns)
        logger.info(
            "History compressed: %d turns → summary + %d recent turns",
            len(history),
            len(recent_turns),
        )
        return compressed

    def condense_with_history(self, query: str, history: list[dict[str, str]]) -> str:
        """
        Turns a follow-up question into a standalone query by using chat history.
        """

        window = self._q_cfg["conversation"]["history_window"]
        model = self._q_cfg["conversation"]["condense_model"]

        trimmed = history[-window:]

        history_str = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in trimmed
        )
        MAX_HISTORY_CHARS = 3_000
        if len(history_str) > MAX_HISTORY_CHARS:
            history_str = "..." + history_str[-MAX_HISTORY_CHARS:]

        system_prompt = (
            "You are a query rewriting engine for a RAG system.\n"
            "Your task is to convert a follow-up question into a standalone question.\n\n"
            "STRICT RULES:\n"
            "1. Use ONLY information explicitly present in the conversation history.\n"
            "2. DO NOT guess, infer, or add missing details.\n"
            "3. If the question is already standalone, return it unchanged but correcting typos if there is any.\n"
            "4. Preserve technical terms exactly as written.\n"
            "5. Do NOT answer the question.\n"
            "6. Output ONLY the rewritten question.\n"
            "7. If the user message is casual conversation, chit-chat, greeting, acknowledgement, "
            "reaction, or small talk (examples: 'ok', 'great', 'thanks', 'interesting', "
            "'hello', 'cool', 'nice'), return the EXACT original user query unchanged.\n"
        )

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Conversation history:\n{history_str}\n\n"
                            f"Follow-up question: {query}\n\n"
                            "Rewritten standalone question:"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip() or query
        except Exception as exc:
            logger.warning("Condense-with-history failed (%s); using raw query.", exc)
            return query

    def _expand_query(self, query: str) -> tuple[str, str | None]:
        """Expand the query using HyDE or query2doc; returns (expanded_query, hypothetical_doc)."""

        method = self._q_cfg["expansion"]["method"]  # "hyde" | "query2doc" | "prf"

        if method == "hyde":
            model = self._q_cfg["expansion"]["hyde_model"]
            max_tok = self._q_cfg["expansion"]["hyde_max_tokens"]
            try:
                response = self._openai.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Before generating, correct any spelling mistakes in the question and "
                                "consider name variations and alternative spellings.\n"
                                "Then write a plausible 2-3 sentence passage as if extracted from a source document "
                                "that directly answers the question in its corrected and variant forms.\n"
                                "Do not refuse. Return ONLY the passage."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    temperature=0.3,
                    max_tokens=max_tok,
                )
                hypothetical_doc = response.choices[0].message.content.strip()

                return f"{query} {hypothetical_doc}", hypothetical_doc
            except Exception as exc:
                logger.warning("HyDE expansion failed (%s); using raw query.", exc)
                return query, None

        elif method == "query2doc":
            try:
                response = self._openai.chat.completions.create(
                    model=self._q_cfg["expansion"]["hyde_model"],
                    messages=[
                        {
                            "role": "user",
                            "content": f"Expand this search query into a detailed question: {query}",
                        }
                    ],
                    temperature=0.0,
                    max_tokens=100,
                )
                expanded = response.choices[0].message.content.strip()
                return f"{query} {expanded}", None
            except Exception as exc:
                logger.warning("query2doc expansion failed (%s); using raw query.", exc)
                return query, None

        else:
            return query, None

    def _classify_routing_intent(
        self,
        query: str,
        ctx_history: list[dict[str, str]],
    ) -> str:
        """
        Classify query intent: "conversational", "followup", or "retrieval".
        Keyword-based, no LLM call.
        """
        q = query.strip().lower()

        CONVERSATIONAL_EXACT = {
            "hi",
            "hello",
            "hey",
            "ok",
            "okay",
            "thanks",
            "thank you",
            "cool",
            "great",
            "nice",
            "sure",
            "interesting",
            "got it",
            "i see",
            "makes sense",
            "sounds good",
            "no problem",
        }
        CONVERSATIONAL_PREFIXES = (
            "how are you",
            "what's up",
            "who are you",
            "are you an ai",
        )
        if q in CONVERSATIONAL_EXACT:
            return "conversational"
        if any(q.startswith(p) for p in CONVERSATIONAL_PREFIXES):
            return "conversational"

        # Phrases that signal a follow-up regardless of whether history exists
        FOLLOWUP_ALWAYS = (
            "you just",
            "you said",
            "you mentioned",
            "i meant",
            "tell me more",
            "expand on",
            "elaborate",
            "clarify",
            "explain more",
            "what do you mean",
            "can you explain",
            "more detail",
            "in more detail",
            "the one you",
            "that algorithm",
            "this algorithm",
            "the algo",
            "those methods",
            "these methods",
            "the methods",
            "the components",
            "that you",
            "this approach",
            "the approach",
        )
        if any(phrase in q for phrase in FOLLOWUP_ALWAYS):
            return "followup"

        # Phrases that only signal a follow-up when there IS prior conversation
        FOLLOWUP_WITH_HISTORY = (
            "summarize it",
            "summarize the last",
            "summarize that",
            "summarize your last",
            "the last answer",
            "the last response",
            "the previous answer",
            "the previous response",
            "what you just said",
            "what did you say",
            "you just said",
            "repeat that",
            "say that again",
            "what about",
        )
        if ctx_history and any(phrase in q for phrase in FOLLOWUP_WITH_HISTORY):
            return "followup"

        PRONOUNS = {"it", "its", "they", "their", "this", "that", "these", "those", "them"}
        SUMMARY_VERBS = {"summarize", "summarise", "recap", "rephrase", "restate", "expand"}
        if ctx_history:
            tokens = set(q.split())
            if tokens & PRONOUNS and len(q.split()) <= 10:
                return "followup"
            if tokens & SUMMARY_VERBS:
                return "followup"

        if len(q.split()) <= 4 and not any(
            kw in q
            for kw in [
                "what",
                "who",
                "why",
                "how",
                "when",
                "where",
                "which",
                "is",
                "are",
                "does",
                "do",
            ]
        ):
            return "conversational"

        return "retrieval"

    def _decompose_query(self, query: str) -> list[str]:
        """
        Decompose a complex multi-intent query into atomic sub-questions.
        """
        max_sub = self._q_cfg["decomposition"]["max_sub_questions"]
        model = self._q_cfg["decomposition"].get(
            "model", self._q_cfg["conversation"]["condense_model"]
        )

        system_prompt = (
            f"You are a question decomposition assistant. "
            f"If the question is already simple and self-contained, return it as a single sub-question. "
            f"If it is complex or multi-hop, break it into at most {max_sub} simpler, "
            f"self-contained sub-questions. Return ONLY a numbered list, one per line."
        )

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Query decomposition failed (%s); skipping.", exc)
            return []

        sub_questions = []
        for line in raw.split("\n"):
            cleaned = re.sub(r"^\d+[.)]\s*", "", line.strip())
            if cleaned:
                sub_questions.append(cleaned)

        return sub_questions[:max_sub]

    def _extract_filters(self, query: str) -> dict[str, Any]:
        """
        Run NER on the query to extract named entities and temporal references.
        These become metadata filters applied during vector store search.
        """
        filters: dict[str, Any] = {}

        if self._nlp:
            doc = self._nlp(query)
            for ent in doc.ents:
                filter_key = f"entity_{ent.label_}"
                filters.setdefault(filter_key, []).append(ent.text)
                logger.debug("NER entity found: %s = '%s'", ent.label_, ent.text)

        if self._q_cfg["entity_recognition"]["temporal_grounding"]:
            date_filter = self._extract_date_filter(query)
            if date_filter:
                filters["date_range"] = date_filter

        return filters

    @staticmethod
    def _extract_date_filter(query: str) -> dict[str, str] | None:
        """
        Extract a date range filter from common temporal expressions in the query.
        Returns a dict with "gte" and "lte" ISO-8601 date strings, or None.
        """

        quarter_match = re.search(r"Q([1-4])\s+(\d{4})", query, re.IGNORECASE)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            year = int(quarter_match.group(2))
            quarter_starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
            quarter_ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
            return {
                "gte": f"{year}-{quarter_starts[quarter]}",
                "lte": f"{year}-{quarter_ends[quarter]}",
            }

        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            year = year_match.group(1)
            return {"gte": f"{year}-01-01", "lte": f"{year}-12-31"}

        return None
