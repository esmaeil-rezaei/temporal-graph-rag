from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import openai

from src.config.settings import get_config, get_secrets
from src.graphrag.schema import (
    EntityNode,
    EntityType,
    ExtractionResult,
    RelationshipEdge,
    RelationshipType,
    entity_type_registry,
)
from src.ingestion.parser import ParsedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a knowledge-graph extraction engine.
Your task is to extract named entities and semantic relationships from the text below.

Rules:
- Return ONLY a single valid JSON object — no markdown, no preamble.
- Normalise entity names: strip honorifics (Mr./Dr./Prof.), use proper case, resolve obvious acronyms.
- Only extract relationships where BOTH ends are present in the entities list.
- Confidence values must be in [0.0, 1.0].
- Use ONLY the entity_type values from this list: {entity_types}
- Use ONLY the relation_type values from this list: {relation_types}

JSON schema (strict):
{{
  "entities": [
    {{
      "name": "<string>",
      "entity_type": "<EntityType>",
      "description": "<one-sentence description or null>",
      "confidence": <float>
    }}
  ],
  "relationships": [
    {{
      "source": "<entity name>",
      "target": "<entity name>",
      "relation_type": "<RelationshipType>",
      "description": "<one-sentence description or null>",
      "weight": <float>
    }}
  ]
}}
"""

_USER_PROMPT = """\
Extract entities and relationships from the following text.

Text:
\"\"\"
{text}
\"\"\"
"""

_HONORIFIC_RE = re.compile(
    r"^\s*(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Sir|Lord|Lady)\s+",
    re.IGNORECASE,
)


def _normalise_name(name: str) -> str:
    """
    Strip honorifics and leading/trailing whitespace.
    Keeps casing (proper nouns should stay proper-cased).
    """
    return _HONORIFIC_RE.sub("", name).strip()


def _safe_entity_type(raw: str) -> EntityType:
    """Map raw LLM label to EntityType, falling back to UNKNOWN."""
    try:
        return EntityType(raw.upper())
    except ValueError:
        logger.debug("Unknown entity type '%s' → UNKNOWN", raw)
        return EntityType.UNKNOWN


def _safe_relation_type(raw: str) -> RelationshipType:
    """Map raw LLM label to RelationshipType, falling back to RELATED_TO."""
    try:
        return RelationshipType(raw.upper())
    except ValueError:
        logger.debug("Unknown relation type '%s' → RELATED_TO", raw)
        return RelationshipType.RELATED_TO


class GraphExtractor:
    """Extracts entities and relationships from text chunks via an async LLM pipeline."""

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._gr_cfg: dict[str, Any] = cfg.get("graphrag", {})
        self._model: str = self._gr_cfg.get("extraction_model", "gpt-4o")
        self._max_chunk_chars: int = self._gr_cfg.get("max_chunk_chars", 4000)
        self._temperature: float = self._gr_cfg.get("extraction_temperature", 0.0)
        self._max_tokens: int = self._gr_cfg.get("extraction_max_tokens", 2048)
        self._concurrency: int = self._gr_cfg.get("extraction_concurrency", 8)
        self._openai = openai.AsyncOpenAI(api_key=sec.openai_api_key)
        self._semaphore: asyncio.Semaphore | None = None

        self._entity_types_str = ", ".join(entity_type_registry.all_labels())
        self._relation_types_str = ", ".join(
            r.value
            for r in RelationshipType
            if r
            not in (
                RelationshipType.MENTIONS,
                RelationshipType.PART_OF,
                RelationshipType.BELONGS_TO_COMMUNITY,
                RelationshipType.CO_OCCURS_WITH,
            )
        )

    def extract(self, chunk: ParsedChunk) -> ExtractionResult:
        """
        Synchronous single-chunk extraction (convenience wrapper).
        Runs the async path in a new event loop.
        """
        loop = asyncio.new_event_loop()
        try:
            sem = asyncio.Semaphore(1)
            return loop.run_until_complete(self._extract_async(chunk, sem))
        finally:
            loop.close()

    async def batch_extract(
        self,
        chunks: list[ParsedChunk],
    ) -> list[ExtractionResult]:
        """Extract entities and relationships from all chunks concurrently, returning one result per chunk."""
        semaphore = asyncio.Semaphore(self._concurrency)
        tasks = [self._extract_async(chunk, semaphore) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed: list[ExtractionResult] = []
        for chunk, result in zip(chunks, results, strict=False):
            if isinstance(result, Exception):
                logger.error(
                    "Extraction failed for chunk %s: %s",
                    chunk.chunk_id,
                    result,
                    exc_info=result,
                )
                processed.append(ExtractionResult(chunk_id=chunk.chunk_id or ""))
            else:
                processed.append(result)

        return processed

    async def _extract_async(
        self, chunk: ParsedChunk, semaphore: asyncio.Semaphore
    ) -> ExtractionResult:
        """Core async extraction for one chunk."""
        async with semaphore:
            chunk_id = chunk.chunk_id or ""
            text = chunk.text[: self._max_chunk_chars]

            system = _SYSTEM_PROMPT.format(
                entity_types=self._entity_types_str,
                relation_types=self._relation_types_str,
            )
            user = _USER_PROMPT.format(text=text)

            t0 = time.monotonic()
            raw_json, prompt_tokens, completion_tokens = await self._call_llm(
                system, user, chunk_id
            )
            latency_ms = (time.monotonic() - t0) * 1000

            parsed = self._parse_response(raw_json, chunk_id)

            entities = self._build_entities(parsed.get("entities", []), chunk_id)
            relationships = self._build_relationships(
                parsed.get("relationships", []), entities, chunk_id
            )

            logger.debug(
                "Extracted %d entities, %d relationships from chunk %s " "(%.0f ms, %d+%d tokens)",
                len(entities),
                len(relationships),
                chunk_id[:12],
                latency_ms,
                prompt_tokens,
                completion_tokens,
            )

            return ExtractionResult(
                chunk_id=chunk_id,
                entities=entities,
                relationships=relationships,
                model_used=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                extraction_latency_ms=latency_ms,
            )

    async def _call_llm(self, system: str, user: str, chunk_id: str) -> tuple[str, int, int]:
        """Call the OpenAI API with exponential-backoff retry; returns (raw_text, prompt_tokens, completion_tokens)."""
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = await self._openai.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
                usage = response.usage
                return (
                    response.choices[0].message.content.strip(),
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
            except openai.RateLimitError as exc:
                wait = 2**attempt
                logger.warning(
                    "RateLimitError on chunk %s attempt %d — retrying in %ds",
                    chunk_id[:12],
                    attempt,
                    wait,
                )
                await asyncio.sleep(wait)
                last_exc = exc
            except Exception as exc:
                logger.warning(
                    "LLM extraction error on chunk %s attempt %d: %s",
                    chunk_id[:12],
                    attempt,
                    exc,
                )
                last_exc = exc
                if attempt < 3:
                    await asyncio.sleep(attempt)

        raise RuntimeError(
            f"LLM extraction failed after 3 attempts for chunk {chunk_id}: {last_exc}"
        )

    def _parse_response(self, raw: str, chunk_id: str) -> dict[str, Any]:
        """Parse LLM JSON output, falling back to regex extraction on decode failure."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(
            "Could not parse extraction JSON for chunk %s — returning empty",
            chunk_id[:12],
        )
        return {"entities": [], "relationships": []}

    def _build_entities(
        self, raw_entities: list[dict[str, Any]], chunk_id: str
    ) -> list[EntityNode]:
        """Convert raw LLM entity dicts into EntityNode instances, skipping malformed entries."""
        nodes: list[EntityNode] = []
        seen_ids: set[str] = set()

        for raw in raw_entities:
            try:
                name = _normalise_name(str(raw.get("name", "")).strip())
                if not name:
                    continue
                entity_type = _safe_entity_type(str(raw.get("entity_type", "UNKNOWN")))
                node = EntityNode(
                    name=name,
                    entity_type=entity_type,
                    description=raw.get("description"),
                    confidence=float(raw.get("confidence", 1.0)),
                    source_chunks=[chunk_id] if chunk_id else [],
                )
                if node.node_id in seen_ids:
                    existing = next(n for n in nodes if n.node_id == node.node_id)
                    if chunk_id and chunk_id not in existing.source_chunks:
                        existing.source_chunks.append(chunk_id)
                    continue
                seen_ids.add(node.node_id)
                nodes.append(node)
            except Exception as exc:
                logger.warning("Skipping malformed entity entry: %s — %s", raw, exc)

        return nodes

    def _build_relationships(
        self,
        raw_relationships: list[dict[str, Any]],
        entities: list[EntityNode],
        chunk_id: str,
    ) -> list[RelationshipEdge]:
        """Convert raw relationship dicts into RelationshipEdge instances, requiring both endpoints in the extracted entity set."""
        name_to_id: dict[str, str] = {_normalise_name(e.name).lower(): e.node_id for e in entities}
        edges: list[RelationshipEdge] = []
        seen_edge_ids: set[str] = set()

        for raw in raw_relationships:
            try:
                source_name = _normalise_name(str(raw.get("source", ""))).lower().strip()
                target_name = _normalise_name(str(raw.get("target", ""))).lower().strip()

                source_id = name_to_id.get(source_name)
                target_id = name_to_id.get(target_name)

                if not source_id or not target_id:
                    logger.debug(
                        "Dropping relationship '%s' → '%s': one or both entities "
                        "not found in extracted set",
                        source_name,
                        target_name,
                    )
                    continue

                relation_type = _safe_relation_type(str(raw.get("relation_type", "RELATED_TO")))
                edge = RelationshipEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    description=raw.get("description"),
                    weight=float(raw.get("weight", 1.0)),
                    source_chunks=[chunk_id] if chunk_id else [],
                )
                if edge.edge_id in seen_edge_ids:
                    existing = next(e for e in edges if e.edge_id == edge.edge_id)
                    existing.weight += edge.weight
                    if chunk_id and chunk_id not in existing.source_chunks:
                        existing.source_chunks.append(chunk_id)
                    continue

                seen_edge_ids.add(edge.edge_id)
                edges.append(edge)

            except Exception as exc:
                logger.warning("Skipping malformed relationship entry: %s — %s", raw, exc)

        return edges
