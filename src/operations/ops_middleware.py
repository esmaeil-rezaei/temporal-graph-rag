from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

import jwt
import numpy as np
import redis
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.config.settings import get_config, get_secrets
from src.generation.generator import GenerationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class SemanticCache:
    """Redis cache for (query_vector → GenerationResult) pairs, using cosine similarity for lookups."""

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._cache_cfg = cfg.operations["performance"]["semantic_cache"]
        self._enabled = self._cache_cfg["enabled"]
        self._threshold = self._cache_cfg["similarity_threshold"]
        self._ttl = self._cache_cfg["ttl_seconds"]

        if self._enabled:
            self._redis = redis.from_url(sec.redis_url)
            self._cache_key_prefix = "rag:semantic_cache:"
            logger.info("Semantic cache initialised (Redis)")

    @staticmethod
    def _namespace_slug(namespace: str | None) -> str:
        """Stable 8-char hex slug for a namespace string (used in cache keys)."""
        return hashlib.sha256((namespace or "default").encode()).hexdigest()[:8]

    def get(
        self,
        query_vector: np.ndarray,
        query_routing_intent: str = "retrieval",
        namespace: str | None = None,
    ) -> GenerationResult | None:
        """Return a cached GenerationResult for the query vector, scoped by namespace; None on miss."""
        if not self._enabled:
            return None

        if query_routing_intent == "conversational":
            logger.debug("Semantic cache skipped (conversational intent)")
            return None

        ns_slug = self._namespace_slug(namespace)
        ns_prefix = f"{self._cache_key_prefix}{ns_slug}:"

        try:
            keys = self._redis.keys(f"{ns_prefix}*")
            for key in keys:
                entry_raw = self._redis.get(key)
                if not entry_raw:
                    continue
                entry = json.loads(entry_raw)
                cached_vec = np.array(entry["vector"], dtype=np.float32)
                similarity = self._cosine_similarity(query_vector, cached_vec)

                if similarity >= self._threshold:
                    logger.info(
                        "Semantic cache HIT (similarity=%.3f, namespace=%s)",
                        similarity,
                        namespace or "default",
                    )
                    return self._deserialise_result(entry["result"])
        except Exception as exc:
            logger.warning("Semantic cache lookup failed: %s", exc)

        return None

    def put(
        self,
        query_vector: np.ndarray,
        result: GenerationResult,
        namespace: str | None = None,
    ) -> None:
        """Store a GenerationResult scoped to a namespace."""
        if not self._enabled:
            return

        ns_slug = self._namespace_slug(namespace)
        vec_hash = hashlib.sha256(query_vector.tobytes()).hexdigest()[:16]
        cache_key = f"{self._cache_key_prefix}{ns_slug}:{vec_hash}"
        entry = {
            "vector": query_vector.tolist(),
            "result": self._serialise_result(result),
        }
        try:
            self._redis.setex(cache_key, self._ttl, json.dumps(entry))
            logger.debug("Semantic cache stored: %s", cache_key)
        except Exception as exc:
            logger.warning("Semantic cache write failed: %s", exc)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1D unit-normalised vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    @staticmethod
    def _serialise_result(result: GenerationResult) -> dict[str, Any]:
        """Convert a GenerationResult to a JSON-serialisable dict."""
        return {
            "answer": result.answer,
            "citations": result.citations,
            "sources": result.sources,
            "faithfulness_score": result.faithfulness_score,
            "has_conflict": result.has_conflict,
            "model_used": result.model_used,
        }

    @staticmethod
    def _deserialise_result(data: dict[str, Any]) -> GenerationResult:
        """Reconstruct a GenerationResult from a cached dict."""
        result = GenerationResult(answer=data["answer"])
        result.citations = data.get("citations", [])
        result.sources = data.get("sources", [])
        result.faithfulness_score = data.get("faithfulness_score")
        result.has_conflict = data.get("has_conflict", False)
        result.model_used = data.get("model_used", "cached")
        return result


class AccessControlMiddleware:
    """
    JWT-based authentication and tenant scoping.
    """

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._acl_cfg = cfg.operations["access_control"]
        self._enabled = self._acl_cfg["enabled"]
        self._jwt_secret = sec.jwt_secret_key or ""

    def authenticate(self, token: str) -> dict[str, Any]:
        if not self._enabled:
            return {"namespace": "default", "roles": ["public"]}

        if not self._jwt_secret:
            raise ValueError("JWT_SECRET_KEY is not configured in .env")

        claims = jwt.decode(
            token,
            self._jwt_secret,
            algorithms=["HS256"],
        )

        claims.setdefault("namespace", "default")
        claims.setdefault("roles", ["public"])
        return claims

    def get_namespace(self, claims: dict[str, Any]) -> str:
        return claims.get("namespace", "default")

    def get_roles(self, claims: dict[str, Any]) -> list[str]:
        return claims.get("roles", ["public"])


class PIIGuard:
    """
    Detects and redacts PII in text.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._pii_cfg = cfg.operations["pii"]
        self._enabled_scan = self._pii_cfg["detect_at_ingestion"]
        self._enabled_input = self._pii_cfg["pii_block_on_input"]
        self._output_scan = self._pii_cfg["output_scanning"]
        self._entities = self._pii_cfg["entities_to_redact"]

        if self._enabled_scan or self._output_scan:
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            logger.info("Presidio PII guard initialised")

    def redact(self, text: str, context: str = "ingestion") -> str:
        """
        Detect and redact PII using configured entity rules.
        """
        if context == "ingestion" and not self._enabled_scan:
            return text
        if context == "output" and not self._output_scan:
            return text
        if context == "query" and not self._enabled_input:
            return text

        try:

            analyzer_results = self._analyzer.analyze(
                text=text,
                entities=self._entities,
                language="en",
            )

            if not analyzer_results:
                return text

            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
            )
            if len(analyzer_results) > 0:
                logger.info(
                    f"PII redacted in {context}: {len(analyzer_results)} entities removed",
                    extra={"entity_types": [r.entity_type for r in analyzer_results]},
                )
            return anonymized.text

        except Exception as exc:
            logger.error(f"PII redaction failed: {exc}")
            return text


class TraceSpan:
    """
    Context manager for request tracing spans.
    """

    def __init__(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        self._name = name
        self._metadata = metadata or {}
        self._span_id = str(uuid.uuid4())[:8]
        self._start_time: float = 0.0

    def __enter__(self) -> TraceSpan:
        self._start_time = time.perf_counter()
        logger.debug(
            f"Span START: {self._name}",
            extra={"span_id": self._span_id, "span_name": self._name, **self._metadata},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        status = "error" if exc_type else "ok"
        logger.info(
            f"Span END: {self._name} [{status}] {elapsed_ms:.1f}ms",
            extra={
                "span_id": self._span_id,
                "span_name": self._name,
                "latency_ms": round(elapsed_ms, 1),
                "status": status,
                "error": str(exc_val) if exc_val else None,
                **self._metadata,
            },
        )


def with_tracing(span_name: str) -> Callable:
    """Wrap a function in a TraceSpan for tracing."""

    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs):
            with TraceSpan(span_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
