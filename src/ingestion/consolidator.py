from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.ingestion.parser import ParsedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConsolidatedChunk:
    """
    Consolidated section-level chunk with merged metadata.
    """

    text: str
    source_file: str
    source_name: str
    modality: str = "text"
    language: str = ""
    doc_version: str | None = None
    ingestion_ts: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    element_metadata: list[dict[str, Any]] = field(default_factory=list)

    def compute_fingerprint(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def chunk_id(self) -> str | None:
        return self.metadata.get("chunk_id")

    @chunk_id.setter
    def chunk_id(self, value: str) -> None:
        self.metadata["chunk_id"] = value


class ChunkConsolidator:
    """
    Merge parser-level chunks into section-level chunks.
    """

    def consolidate(self, parsed_chunks: list[ParsedChunk]) -> list[ConsolidatedChunk]:
        """
        Group parser chunks into consolidated section chunks.
        """
        if not parsed_chunks:
            return []

        groups: dict[str, list[ParsedChunk]] = {}
        group_order: list[str] = []

        for chunk in parsed_chunks:
            key = self._section_key(chunk)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(chunk)

        consolidated: list[ConsolidatedChunk] = []
        for key in group_order:
            merged = self._merge_group(groups[key])
            if merged.text.strip():
                consolidated.append(merged)

        logger.info(f"Consolidation: {len(parsed_chunks)} elements → {len(consolidated)} sections")
        return consolidated

    @staticmethod
    def _section_key(chunk: ParsedChunk) -> str:
        """
        Build the grouping key for section consolidation.
        """
        section = chunk.metadata.get("section") or chunk.metadata.get("breadcrumb") or ""

        return f"{chunk.source_file}::{section}"

    def _merge_group(self, chunks: list[ParsedChunk]) -> ConsolidatedChunk:
        """
        Merge same-section chunks into one consolidated chunk.
        """
        text_parts: list[str] = []
        all_metadata: list[dict[str, Any]] = [c.metadata for c in chunks]

        for chunk in chunks:
            category = chunk.metadata.get("category", "")
            part = self._format_element(
                chunk.text.strip(), category, chunk.metadata.get("section_depth", 2)
            )
            if part:
                text_parts.append(part)

        merged_text = "\n\n".join(p for p in text_parts if p.strip())

        return ConsolidatedChunk(
            text=merged_text,
            source_file=chunks[0].source_file,
            source_name=chunks[0].source_name,
            modality=self._dominant_modality(chunks),
            language=self._dominant_language(chunks),
            doc_version=chunks[0].doc_version,
            ingestion_ts=chunks[0].ingestion_ts,
            metadata=self._merge_metadata(all_metadata),
            element_metadata=all_metadata,
        )

    @staticmethod
    def _format_element(text: str, category: str, depth: int) -> str:
        """
        Format chunk text based on element category.
        """
        if not text:
            return ""

        if category == "Title":
            # Recreate Markdown headings for hierarchical section splitting.
            hashes = "#" * max(1, min(depth, 6))
            return f"{hashes} {text}"

        elif category == "Table":
            return f"[TABLE]\n{text}\n[/TABLE]"

        elif category == "Image":
            return f"[IMAGE_DESCRIPTION]\n{text}\n[/IMAGE_DESCRIPTION]"

        elif category in {"ListItem", "ListItem.Bulleted", "ListItem.Numbered"}:
            return f"- {text}" if not text.startswith("-") else text

        else:
            return text

    @staticmethod
    def _merge_metadata(all_meta: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Merge element metadata into section-level metadata.
        """
        if not all_meta:
            return {}

        merged: dict[str, Any] = {}

        for key in (
            "section",
            "section_depth",
            "filename",
            "breadcrumb",
            "front_matter",
            "suffix",
            "doc_version",
        ):
            for m in all_meta:
                val = m.get(key)
                if val is not None and val != "":
                    merged[key] = val
                    break

        pages = [m["page_number"] for m in all_meta if isinstance(m.get("page_number"), int)]
        if pages:
            merged["page_start"] = min(pages)
            merged["page_end"] = max(pages)

        categories = list({m["category"] for m in all_meta if m.get("category")})
        merged["categories"] = categories
        merged["has_table"] = "Table" in categories
        merged["has_image"] = "Image" in categories
        merged["has_list"] = any(c.startswith("ListItem") for c in categories)

        merged["element_ids"] = [m["element_id"] for m in all_meta if m.get("element_id")]
        merged["element_count"] = len(all_meta)

        return merged

    @staticmethod
    def _dominant_language(chunks: list[ParsedChunk]) -> str:
        """
        Return the majority language across all non-None language values.
        """
        langs = [c.language for c in chunks if c.language]
        if not langs:
            return ""
        return Counter(langs).most_common(1)[0][0]

    @staticmethod
    def _dominant_modality(chunks: list[ParsedChunk]) -> str:
        """
        Return the dominant modality for a consolidated chunk.
        """
        modalities = {c.modality for c in chunks}
        if modalities == {"table"}:
            return "table"
        if modalities == {"image_caption"}:
            return "image_caption"
        return "text"
