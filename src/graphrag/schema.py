from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.config.settings import get_config

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """
    Infrastructure entity labels that the graph layer depends on directly.

    DO NOT add domain labels here.  Register them via EntityTypeRegistry
    at application startup instead.

    ``UNKNOWN`` is the catch-all when the LLM returns a label that is not
    in the registered set.  It is never written to the prompt — it exists
    only as a Python fallback so callers always get a valid EntityType back.
    """

    CHUNK = "CHUNK"  # every ingested text chunk as a node
    DOCUMENT = "DOCUMENT"  # source-document root node
    COMMUNITY = "COMMUNITY"  # community cluster node
    UNKNOWN = "UNKNOWN"  # catch-all for unrecognised LLM output


class RelationshipType(str, Enum):
    """
    Structural relationship labels that the graph layer depends on directly.

    DO NOT add domain-specific semantic relations here.  Register them via
    RelationshipTypeRegistry at application startup instead.

    ``RELATED_TO`` is the generic fallback for LLM-extracted relations that
    don't match any registered label.
    """

    MENTIONS = "MENTIONS"  # Chunk → Entity
    PART_OF = "PART_OF"  # Chunk → Document
    BELONGS_TO_COMMUNITY = "BELONGS_TO_COMMUNITY"  # Entity → Community
    CO_OCCURS_WITH = "CO_OCCURS_WITH"  # Entity ↔ Entity (co-occurrence)

    RELATED_TO = "RELATED_TO"  # catch-all for unrecognised LLM output


class EntityTypeRegistry:
    """
    Open-world registry for domain entity type labels.

    At startup, call ``EntityTypeRegistry.load_from_config()`` once.
    Afterward, ``resolve(label)`` converts any raw LLM string to either
    a registered label string or ``EntityType.UNKNOWN``.

    The registry is a module-level singleton accessed via
    ``entity_type_registry``.

    Thread safety: registrations should happen at startup before any
    concurrent extraction runs.  Post-startup registrations are supported
    but callers are responsible for synchronisation if needed.

    Design rationale
    ----------------
    We deliberately do NOT extend the EntityType enum at runtime (the old
    approach in v1).  Dynamic enum mutation is fragile: it bypasses the
    metaclass, confuses type checkers, and breaks pickle/deepcopy.  Instead
    we keep the enum small and load-bearing, and use this registry for
    everything domain-specific.  The extractor's prompt builder reads
    ``entity_type_registry.all_labels()`` so the LLM always sees the
    full current set.
    """

    # Default seed labels that cover the majority of general-purpose deployments.
    # These are only used if config.yaml does not specify entity_types at all.
    _DEFAULT_SEEDS: frozenset[str] = frozenset(
        {
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "DATE",
            "EVENT",
            "PRODUCT",
            "TECHNOLOGY",
            "CONCEPT",
        }
    )

    def __init__(self) -> None:
        # Start with infrastructure labels always present.
        self._labels: set[str] = {e.value for e in EntityType}

    def register(self, *labels: str) -> None:
        """
        Register one or more domain entity type labels.

        Labels are upper-cased and stripped automatically.
        Duplicate registrations are silently ignored.
        Infrastructure labels (CHUNK, DOCUMENT, COMMUNITY, UNKNOWN)
        are always present and cannot be removed.

        Example::

            entity_type_registry.register("GENE", "DRUG", "CLINICAL_TRIAL")
        """
        for raw in labels:
            label = raw.strip().upper()
            if not label:
                continue
            if label not in self._labels:
                self._labels.add(label)
                logger.debug("EntityTypeRegistry: registered '%s'", label)

    def load_from_config(self, config_entity_types: list[str] | None) -> None:
        """
        Bulk-register labels from the ``graphrag.entity_types`` config list.

        If the list is empty or None, the default seed labels are loaded
        so a freshly deployed system has a reasonable starting set without
        requiring any config changes.

        Call once at application startup::

            from src.config.settings import get_config
            cfg = get_config()
            entity_type_registry.load_from_config(
                cfg.get("graphrag.entity_types")
            )
        """
        labels = config_entity_types or list(self._DEFAULT_SEEDS)
        self.register(*labels)
        logger.info(
            "EntityTypeRegistry: %d labels loaded (%d total including infrastructure)",
            len(labels),
            len(self._labels),
        )

    def resolve(self, raw_label: str) -> str:
        """
        Normalise a raw LLM label to a registered string.

        Returns the label unchanged if it is registered (after upper-casing).
        Returns ``EntityType.UNKNOWN.value`` for any unrecognised label so
        callers always get a valid, safe string back.

        The extractor uses this to validate every entity type returned by
        the LLM before building an EntityNode.
        """
        normalised = raw_label.strip().upper()
        if normalised in self._labels:
            return normalised
        logger.debug("EntityTypeRegistry: unrecognised label '%s' → UNKNOWN", raw_label)
        return EntityType.UNKNOWN.value

    def all_labels(self) -> list[str]:
        """
        Return all registered labels sorted alphabetically.

        Used by the extractor to build the system prompt so the LLM
        only generates types that exist in the registry.
        Infrastructure-only labels (CHUNK, DOCUMENT, COMMUNITY) are
        excluded from the prompt because the LLM should never extract
        those — they are structural.
        """
        _prompt_excluded = {
            EntityType.CHUNK.value,
            EntityType.DOCUMENT.value,
            EntityType.COMMUNITY.value,
            EntityType.UNKNOWN.value,
        }
        return sorted(self._labels - _prompt_excluded)

    def is_registered(self, label: str) -> bool:
        return label.strip().upper() in self._labels

    def __len__(self) -> int:
        return len(self._labels)

    def __repr__(self) -> str:
        return f"EntityTypeRegistry({sorted(self._labels)})"


class RelationshipTypeRegistry:
    """
    Open-world registry for domain relationship type labels.

    Mirrors EntityTypeRegistry but for edges.  Structural edges
    (MENTIONS, PART_OF, BELONGS_TO_COMMUNITY, CO_OCCURS_WITH, RELATED_TO)
    are always present.  Domain-specific semantic relations are loaded
    from ``config.yaml → graphrag.relationship_types``.

    Default seeds cover general-purpose deployments; override entirely
    in config for domain-specific systems.
    """

    _DEFAULT_SEEDS: frozenset[str] = frozenset(
        {
            "WORKS_FOR",
            "LOCATED_IN",
            "FOUNDED_BY",
            "OWNS",
            "PRODUCES",
            "CAUSES",
            "INTERACTS_WITH",
            "REGULATES",
            "CITES",
            "SUCCEEDED_BY",
            "PRECEDED_BY",
            "HAPPENED_ON",
            "STARTED_ON",
            "ENDED_ON",
        }
    )

    def __init__(self) -> None:
        # Structural labels always present.
        self._labels: set[str] = {r.value for r in RelationshipType}

    def register(self, *labels: str) -> None:
        """Register one or more semantic relationship type labels."""
        for raw in labels:
            label = raw.strip().upper()
            if label and label not in self._labels:
                self._labels.add(label)
                logger.debug("RelationshipTypeRegistry: registered '%s'", label)

    def load_from_config(self, config_rel_types: list[str] | None) -> None:
        """
        Bulk-register labels from ``graphrag.relationship_types``.
        Falls back to default seeds if list is empty/None.
        """
        labels = config_rel_types or list(self._DEFAULT_SEEDS)
        self.register(*labels)
        logger.info(
            "RelationshipTypeRegistry: %d labels loaded (%d total including structural)",
            len(labels),
            len(self._labels),
        )

    def resolve(self, raw_label: str) -> str:
        """
        Normalise a raw LLM label.
        Returns RELATED_TO for any unrecognised label.
        """
        normalised = raw_label.strip().upper()
        if normalised in self._labels:
            return normalised
        logger.debug(
            "RelationshipTypeRegistry: unrecognised label '%s' → RELATED_TO",
            raw_label,
        )
        return RelationshipType.RELATED_TO.value

    def all_labels(self) -> list[str]:
        """
        All labels except purely structural ones that the LLM should never produce.
        """
        _prompt_excluded = {
            RelationshipType.MENTIONS.value,
            RelationshipType.PART_OF.value,
            RelationshipType.BELONGS_TO_COMMUNITY.value,
            RelationshipType.CO_OCCURS_WITH.value,
        }
        return sorted(self._labels - _prompt_excluded)

    def is_registered(self, label: str) -> bool:
        return label.strip().upper() in self._labels

    def __len__(self) -> int:
        return len(self._labels)

    def __repr__(self) -> str:
        return f"RelationshipTypeRegistry({sorted(self._labels)})"


#: Global entity type registry.  Populated at startup via
#: ``entity_type_registry.load_from_config(...)``.
entity_type_registry = EntityTypeRegistry()

#: Global relationship type registry.  Populated at startup via
#: ``relationship_type_registry.load_from_config(...)``.
relationship_type_registry = RelationshipTypeRegistry()


def bootstrap_registries() -> None:
    """
    Load entity and relationship type registries from config.yaml.

    Call this once at application startup — in ``IngestionPipeline.__init__``,
    ``GraphRetriever.__init__``, or ``app/main.py`` startup — before any
    extraction or retrieval runs.

    Idempotent: safe to call multiple times (duplicate labels are ignored).

    Example::

        from src.graphrag.schema import bootstrap_registries
        bootstrap_registries()
    """
    try:
        cfg = get_config()
        gr_cfg: dict[str, Any] = cfg.get("graphrag", {})

        entity_type_registry.load_from_config(gr_cfg.get("entity_types"))  # None → use defaults
        relationship_type_registry.load_from_config(
            gr_cfg.get("relationship_types")  # None → use defaults
        )
        logger.info(
            "GraphRAG registries bootstrapped: %d entity types, %d relationship types",
            len(entity_type_registry),
            len(relationship_type_registry),
        )
    except Exception as exc:
        logger.error("Registry bootstrap failed — falling back to seed defaults: %s", exc)
        entity_type_registry.load_from_config(None)
        relationship_type_registry.load_from_config(None)


@dataclass
class EntityNode:
    """
    Represents a named entity or concept node in the knowledge graph.

    ``entity_type`` is now a plain string (validated against the registry)
    rather than an EntityType enum member.  This removes the tight coupling
    between domain concepts and the Python enum while preserving all the
    type-safety guarantees we care about: the registry ensures only known
    labels reach Neo4j, and UNKNOWN is the explicit fallback.

    ``node_id`` is a deterministic SHA-256 fingerprint of
    ``(entity_type, normalised_name)`` so identical entities extracted from
    different chunks resolve to the same graph node via MERGE.
    """

    name: str
    entity_type: str  # validated registry label string

    description: str | None = None
    source_chunks: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    node_id: str = field(init=False)

    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        # Normalise the type through the registry so callers don't have to.
        self.entity_type = entity_type_registry.resolve(self.entity_type)
        self.node_id = self._compute_id(self.entity_type, self.name)

    @staticmethod
    def _compute_id(entity_type: str, name: str) -> str:
        """Deterministic SHA-256 fingerprint for idempotent MERGE."""
        key = f"{entity_type}::{name.strip().lower()}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def touch(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()


@dataclass
class RelationshipEdge:
    """
    A directed edge between two EntityNode instances.

    ``relation_type`` is a plain string validated against
    RelationshipTypeRegistry.  Unknown types fall back to RELATED_TO.

    ``edge_id`` is a deterministic fingerprint of
    ``(source_id, relation_type, target_id)`` — idempotent MERGE-safe.
    """

    source_id: str
    target_id: str
    relation_type: str  # validated registry label string

    description: str | None = None
    weight: float = 1.0
    source_chunks: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    edge_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.relation_type = relationship_type_registry.resolve(self.relation_type)
        self.edge_id = self._compute_id(self.source_id, self.relation_type, self.target_id)

    @staticmethod
    def _compute_id(source_id: str, relation_type: str, target_id: str) -> str:
        key = f"{source_id}::{relation_type}::{target_id}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def touch(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()


@dataclass
class CommunityNode:
    """
    A community (cluster) of semantically related entities.

    Detected by Louvain / Leiden on the entity co-occurrence graph and
    stored as a first-class node in Neo4j.  The LLM-generated summary
    enables global thematic retrieval.
    """

    community_id: str
    level: int
    member_ids: list[str]
    summary: str | None = None
    embedding: list[float] | None = None
    title: str | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ExtractionResult:
    """
    Return type of GraphExtractor.extract() / batch_extract().

    One instance per input chunk.  Empty entities/relationships indicate
    either no extractable content or a failed extraction.
    """

    chunk_id: str
    entities: list[EntityNode] = field(default_factory=list)
    relationships: list[RelationshipEdge] = field(default_factory=list)

    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extraction_latency_ms: float = 0.0
