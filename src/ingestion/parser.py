from __future__ import annotations

import base64
import hashlib
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import camelot
import openai
from langdetect import detect as detect_language
from unstructured.documents.elements import Element, Image, Table, Text, Title
from unstructured.partition.auto import partition
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf

from src.config.settings import get_config, get_secrets
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedChunk:
    """
    Metadata contract for the consolidator
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None
    source_file: str | None = None
    source_name: str | None = None
    modality: str = "text"
    language: str | None = None
    doc_version: str | None = None
    ingestion_ts: str | None = None

    def compute_fingerprint(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class _HeadingTracker:
    """
    Maintains a heading stack for hierarchical section tracking.
    Invariant: stack is ordered by increasing heading depth.
    """

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def push(self, depth: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= depth:
            self._stack.pop()
        self._stack.append((depth, title))

    @property
    def section(self) -> str:
        """Text of the nearest heading (leaf of the stack)."""
        return self._stack[-1][1] if self._stack else ""

    @property
    def section_depth(self) -> int:
        """Numeric depth of the nearest heading."""
        return self._stack[-1][0] if self._stack else 0

    @property
    def breadcrumb(self) -> str:
        """Full ancestor path, e.g. 'Introduction > Background > Methods'."""
        return " > ".join(title for _, title in self._stack)


def _infer_depth_from_font_size(
    title_elements: list[Element],
    max_levels: int = 3,
) -> dict[int, int]:
    """
    Infer heading depth from visual layout when metadata is missing.

    Heights of Title elements are used to approximate relative heading levels
    within a document (not absolute font sizes).

    Returns:
        {element_id: depth}
    """

    heights: dict[int, float] = {}
    for el in title_elements:
        meta = getattr(el, "metadata", None)
        coords = getattr(meta, "coordinates", None)
        if coords is None:
            continue
        pts = getattr(coords, "points", None)
        if pts and len(pts) >= 2:
            ys = [p[1] for p in pts]
            heights[id(el)] = max(ys) - min(ys)

    if not heights:
        return {}

    unique_sorted = sorted(set(heights.values()), reverse=True)

    n_levels = min(len(unique_sorted), max_levels)
    if n_levels == 0:
        return {}
    bucket_size = (unique_sorted[0] - unique_sorted[-1] + 1) / n_levels
    depth_map: dict[int, int] = {}
    for eid, h in heights.items():
        bucket = int((unique_sorted[0] - h) / max(bucket_size, 1e-9))
        depth_map[eid] = min(bucket + 1, max_levels)  # 1-indexed depth
    return depth_map


class DocumentParser:
    """
    Routes files to the appropriate parser and returns normalized ParsedChunks.
    Chunks include section metadata for consistent downstream consolidation.
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._sec = get_secrets()
        self._ingest_cfg = self._cfg.ingestion
        self._openai = openai.OpenAI(api_key=self._sec.openai_api_key)

    def parse_file(self, file_path: Path, source_name: str) -> list[ParsedChunk]:
        logger.info("Parsing file", extra={"file": str(file_path), "source": source_name})
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(file_path, source_name)
        elif suffix == ".md":
            return self._parse_md(file_path, source_name)
        elif suffix == ".txt":
            return self._parse_txt(file_path, source_name)
        elif suffix == ".docx":
            return self._parse_docx(file_path, source_name)
        elif suffix == ".html":
            return self._parse_html(file_path, source_name)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            return self._parse_standalone_image(file_path, source_name)
        else:
            logger.warning(f"Unsupported extension: {suffix} for {file_path}")
            return []

    def _parse_pdf(self, file_path: Path, source_name: str) -> list[ParsedChunk]:
        """
        Parse PDF into chunked elements with hierarchical section metadata.
        Section context is derived using metadata when available, otherwise layout heuristics.
        """
        parsing_cfg = self._ingest_cfg["parsing"]

        # Use a per-document subdirectory so images from different PDFs don't collide.
        base_image_dir = Path(parsing_cfg["image_output_dir"])
        doc_image_dir = base_image_dir / file_path.stem
        doc_image_dir.mkdir(parents=True, exist_ok=True)

        elements: list[Element] = partition_pdf(
            filename=str(file_path),
            strategy=parsing_cfg["pdf_strategy"],
            languages=parsing_cfg["ocr_languages"],
            extract_images_in_pdf=parsing_cfg["extract_images"],
            extract_image_block_output_dir=str(doc_image_dir),
        )

        title_elements = [el for el in elements if isinstance(el, Title)]
        fallback_depth_map = _infer_depth_from_font_size(title_elements)

        tracker = _HeadingTracker()
        chunks: list[ParsedChunk] = []
        # Map page_number → tracker snapshot, updated as we walk elements.
        # Used to assign correct section context to Camelot tables by page.
        page_context: dict[int, dict[str, Any]] = {}

        for element in elements:
            # Record the current tracker state for this element's page.
            page_num = getattr(getattr(element, "metadata", None), "page_number", None)
            if page_num is not None:
                page_context.setdefault(
                    page_num,
                    {
                        "section": tracker.section,
                        "section_depth": tracker.section_depth,
                        "breadcrumb": tracker.breadcrumb,
                    },
                )

            if isinstance(element, Title):
                meta = getattr(element, "metadata", None)
                depth = (
                    getattr(meta, "category_depth", None)
                    or fallback_depth_map.get(id(element))
                    or 1
                )
                chunk = self._element_to_chunk(element, file_path, source_name, "text")
                self._attach_section_context(chunk, tracker)
                chunks.append(chunk)
                tracker.push(int(depth), element.text or "")

            elif isinstance(element, Text):
                chunk = self._element_to_chunk(element, file_path, source_name, "text")
                self._attach_section_context(chunk, tracker)
                chunks.append(chunk)

            elif isinstance(element, Table):
                if not self._ingest_cfg["tables"]["extract_tables"]:
                    chunk = self._element_to_chunk(element, file_path, source_name, "table")
                    self._attach_section_context(chunk, tracker)
                    chunks.append(chunk)

            elif isinstance(element, Image):
                if self._ingest_cfg["image_captioning"]["enabled"]:
                    caption = self._caption_image(element, file_path, source_name)
                    if caption:
                        self._attach_section_context(caption, tracker)
                        chunks.append(caption)

        if self._ingest_cfg["tables"]["extract_tables"]:
            camelot_chunks = self._extract_tables_camelot(file_path, source_name)
            for c in camelot_chunks:
                page = c.metadata.get("page_number")
                ctx = page_context.get(page, {})
                c.metadata.setdefault("section", ctx.get("section", ""))
                c.metadata.setdefault("section_depth", ctx.get("section_depth", 0))
                c.metadata.setdefault("breadcrumb", ctx.get("breadcrumb", ""))
            chunks.extend(camelot_chunks)

        return chunks

    def _parse_md(self, file_path: Path, source_name: str) -> list[ParsedChunk]:
        """
        Parse Markdown using unstructured's partition_md with heading-tracker.
        Output shape matches _parse_pdf exactly.
        """
        text = _read_with_encoding_fallback(
            file_path, self._cfg.knowledge_base.get("encoding", "utf-8")
        )
        elements: list[Element] = partition_md(text=text)

        tracker = _HeadingTracker()
        chunks: list[ParsedChunk] = []

        for element in elements:
            if isinstance(element, Title):
                depth = getattr(getattr(element, "metadata", None), "category_depth", 1) or 1
                chunk = self._element_to_chunk(element, file_path, source_name, "text")
                if isinstance(element, Table):
                    chunk.modality = "table"
                self._attach_section_context(chunk, tracker)
                chunks.append(chunk)
                tracker.push(int(depth), element.text or "")
                continue

            chunk = self._element_to_chunk(element, file_path, source_name, "text")
            if isinstance(element, Table):
                chunk.modality = "table"
            self._attach_section_context(chunk, tracker)
            chunks.append(chunk)

        return chunks

    def _parse_txt(self, file_path: Path, source_name: str) -> list[ParsedChunk]:
        """
        Parse .txt as double-newline-separated paragraphs.
        No headings available — all chunks share section="" and depth=0.
        """
        text = _read_with_encoding_fallback(
            file_path, self._cfg.knowledge_base.get("encoding", "utf-8")
        )
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: list[ParsedChunk] = []
        for i, para in enumerate(paragraphs):
            chunk = ParsedChunk(
                text=para,
                modality="text",
                source_file=str(file_path),
                source_name=source_name,
                metadata={
                    "category": "NarrativeText",
                    "element_id": None,
                    "section": "",
                    "section_depth": 0,
                    "breadcrumb": "",
                    "page_number": None,
                    "paragraph_index": i,
                },
            )
            chunk.language = self._detect_language(para)
            chunk.chunk_id = chunk.compute_fingerprint()
            chunks.append(chunk)

        return chunks

    def _extract_tables_camelot(self, file_path: Path, source_name: str) -> list[ParsedChunk]:
        """
        Extract tables from a PDF using Camelot.
        Returns raw table chunks without section metadata.
        """
        chunks: list[ParsedChunk] = []
        output_fmt = self._ingest_cfg["tables"]["output_format"]

        try:
            tables = camelot.read_pdf(str(file_path), pages="all", flavor="lattice")
        except Exception as exc:
            logger.warning(f"Camelot failed for {file_path}: {exc}")
            return chunks

        for i, table in enumerate(tables):
            df = table.df
            raw_header = df.iloc[0]
            col_metadata = [
                col.split("\n")[0] if "\n" in str(col) else str(col) for col in raw_header
            ]

            if output_fmt == "json":
                records = df.iloc[1:].to_dict(orient="records")
                text_repr = str(records)
            else:
                df_body = df.iloc[1:].copy()
                df_body.columns = col_metadata
                text_repr = df_body.to_markdown(index=False)

            chunk = ParsedChunk(
                text=text_repr,
                modality="table",
                source_file=str(file_path),
                source_name=source_name,
                metadata={
                    "category": "Table",
                    "element_id": None,
                    "table_index": i,
                    "page_number": table.page,
                    "columns": col_metadata,
                    "accuracy": table.accuracy,
                    "whitespace": table.whitespace,
                    "section": "",
                    "section_depth": 0,
                    "breadcrumb": "",
                },
            )
            chunk.chunk_id = chunk.compute_fingerprint()
            chunks.append(chunk)

        return chunks

    def _caption_image(
        self, element: Image, file_path: Path, source_name: str
    ) -> ParsedChunk | None:
        """
        Generate a caption for an image using GPT-4V when image data is available.
        """
        image_path: str | None = getattr(getattr(element, "metadata", None), "image_path", None)
        if not image_path:
            logger.debug("Image element has no image_path — skipping captioning")
            return None

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except OSError as exc:
            logger.warning(f"Cannot read image file {image_path}: {exc}")
            return None

        if not image_bytes:
            return None

        mime_type, _ = mimetypes.guess_type(image_path)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type or 'image/png'};base64,{b64}"

        model = self._ingest_cfg["image_captioning"]["model"]
        max_tok = self._ingest_cfg["image_captioning"]["max_tokens"]

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                            {
                                "type": "text",
                                "text": (
                                    "Describe this figure or chart in detail, including all "
                                    "axis labels, data series, trends, and key values visible."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=max_tok,
            )
        except Exception as exc:
            logger.error(f"VLM captioning failed: {exc}")
            return None

        caption: str = response.choices[0].message.content or ""

        chunk = ParsedChunk(
            text=caption,
            modality="image_caption",
            source_file=str(file_path),
            source_name=source_name,
            metadata={
                "category": "Image",
                "element_id": getattr(element, "id", None),
                "page_number": getattr(getattr(element, "metadata", None), "page_number", None),
                "section": "",  # section/breadcrumb filled by caller
                "section_depth": 0,
                "breadcrumb": "",
                "has_clip_embedding": False,
            },
        )
        chunk.chunk_id = chunk.compute_fingerprint()
        return chunk

    def _parse_standalone_image(self, file_path: Path, source_name: str) -> list[ParsedChunk]:
        with open(file_path, "rb") as fh:
            image_bytes = fh.read()

        mime_type, _ = mimetypes.guess_type(str(file_path))
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type or 'image/png'};base64,{b64}"

        model = self._ingest_cfg["image_captioning"]["model"]
        max_tok = self._ingest_cfg["image_captioning"]["max_tokens"]

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                            {"type": "text", "text": "Describe this image in full detail."},
                        ],
                    }
                ],
                max_tokens=max_tok,
            )
        except Exception as exc:
            logger.error(f"Standalone image captioning failed for {file_path}: {exc}")
            return []

        caption = response.choices[0].message.content or ""
        chunk = ParsedChunk(
            text=caption,
            modality="image_caption",
            source_file=str(file_path),
            source_name=source_name,
            metadata={
                "category": "Image",
                "element_id": None,
                "original_filename": file_path.name,
                "section": "",
                "section_depth": 0,
                "breadcrumb": "",
            },
        )
        chunk.chunk_id = chunk.compute_fingerprint()
        return [chunk]

    def _parse_docx(self, file_path: Path, source_name: str) -> list[ParsedChunk]:
        elements: list[Element] = partition(filename=str(file_path), strategy="fast")
        return self._walk_elements_with_tracker(elements, file_path, source_name)

    def _parse_html(self, file_path: Path, source_name: str) -> list[ParsedChunk]:
        elements: list[Element] = partition(filename=str(file_path), strategy="fast")
        return self._walk_elements_with_tracker(elements, file_path, source_name)

    def _walk_elements_with_tracker(
        self,
        elements: list[Element],
        file_path: Path,
        source_name: str,
    ) -> list[ParsedChunk]:
        """
        Apply heading-tracker logic to DOCX/HTML elements to assign section context.
        """
        tracker = _HeadingTracker()
        chunks: list[ParsedChunk] = []

        for element in elements:
            if not isinstance(element, Text | Title | Table):
                continue
            if isinstance(element, Title):
                meta = getattr(element, "metadata", None)
                depth = getattr(meta, "category_depth", 1) or 1
                chunk = self._element_to_chunk(element, file_path, source_name, "text")
                self._attach_section_context(chunk, tracker)
                chunks.append(chunk)
                tracker.push(int(depth), element.text or "")
                continue

            modality = "table" if isinstance(element, Table) else "text"
            chunk = self._element_to_chunk(element, file_path, source_name, modality)
            self._attach_section_context(chunk, tracker)
            chunks.append(chunk)

        return chunks

    def _element_to_chunk(
        self,
        element: Element,
        file_path: Path,
        source_name: str,
        modality: str,
    ) -> ParsedChunk:
        text = element.text or ""
        meta = getattr(element, "metadata", None)
        category = type(element).__name__

        if isinstance(element, Table):
            modality = "table"

        chunk = ParsedChunk(
            text=text,
            modality=modality,
            source_file=str(file_path),
            source_name=source_name,
            metadata={
                "category": category,
                "element_id": getattr(element, "id", None),
                "page_number": getattr(meta, "page_number", None),
                "coordinates": getattr(meta, "coordinates", None),
                "section": "",
                "section_depth": 0,
                "breadcrumb": "",
            },
        )
        if modality == "text":
            chunk.language = self._detect_language(text)
        chunk.chunk_id = chunk.compute_fingerprint()
        return chunk

    @staticmethod
    def _attach_section_context(chunk: ParsedChunk, tracker: _HeadingTracker) -> None:
        """
        Stamp the current heading-tracker state onto a chunk's metadata.
        Called immediately after creating the chunk, before appending.
        """
        chunk.metadata["section"] = tracker.section
        chunk.metadata["section_depth"] = tracker.section_depth
        chunk.metadata["breadcrumb"] = tracker.breadcrumb

    @staticmethod
    def _detect_language(text: str) -> str | None:
        if len(text.strip()) < 20:
            return None
        try:
            return detect_language(text)
        except Exception:
            return None


def _read_with_encoding_fallback(file_path: Path, preferred: str) -> str:
    """
    Read text file using fallback encodings for compatibility across sources.
    """
    for encoding in [preferred, "utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return file_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return file_path.read_bytes().decode("utf-8", errors="replace")
