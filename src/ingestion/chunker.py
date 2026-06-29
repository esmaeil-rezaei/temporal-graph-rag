from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

import numpy as np
import tiktoken
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import get_config, get_secrets
from src.ingestion.parser import ParsedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Matches an ATX heading
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Honorifics/abbreviations that end with a period but are NOT sentence endings.
_ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.$",
    re.IGNORECASE,
)

# Candidate split: a period/!/? followed by whitespace then an uppercase letter
# or digit/quote/bracket.  Fixed-width lookbehind only — Python re compatible.
_CANDIDATE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _is_sentence_boundary(left: str) -> bool:
    """
    Check whether a candidate split is a real sentence boundary.
    """
    left = left.rstrip()
    if _ABBREV.search(left):
        return False
    return True


def _split_on_boundaries(text: str) -> list[str]:
    """
    Split text using validated sentence boundaries.
    """
    sentences: list[str] = []
    prev = 0
    for match in _CANDIDATE_SPLIT.finditer(text):
        left_fragment = text[prev : match.start()]
        if _is_sentence_boundary(left_fragment):
            sentences.append(left_fragment.strip())
            prev = match.end()

    tail = text[prev:].strip()
    if tail:
        sentences.append(tail)
    return sentences if sentences else [text.strip()]


@dataclass
class ChunkNode:
    """
    Chunk node with hierarchical metadata.
    """

    chunk: ParsedChunk
    level: str = "paragraph"  # document | section | paragraph
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)


class TextChunker:
    """
    Chunk documents for embedding and retrieval.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._chunk_cfg = cfg.chunking
        self._emb_cfg = cfg.embeddings
        self._tokeniser = tiktoken.get_encoding("cl100k_base")
        self._embedder = None

    def _get_embedder(self):
        """Initialize the embedding backend using the shared embeddings config."""
        if self._embedder is not None:
            return self._embedder

        backend = self._emb_cfg.get("backend", "sentence_transformers")
        model_name = self._emb_cfg.get("default_model", "BAAI/bge-large-en-v1.5")

        if backend == "openai":
            client = OpenAI(api_key=get_secrets().openai_api_key)

            def embed(texts: list[str]) -> np.ndarray:
                response = client.embeddings.create(input=texts, model=model_name)
                return np.array([item.embedding for item in response.data], dtype=np.float32)

            self._embedder = embed

        elif backend == "sentence_transformers":
            st_model = SentenceTransformer(model_name)

            def embed(texts: list[str]) -> np.ndarray:
                return st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

            self._embedder = embed

        else:
            raise ValueError(f"Unknown embedding_backend: {backend!r}")

        return self._embedder

    def chunk(self, document_chunks: list[ParsedChunk]) -> list[ChunkNode]:
        """
        Accept a list of Consolidated/ParsedChunks and return a flat list of
        ChunkNodes ready for embedding and indexing.
        """
        strategy = self._chunk_cfg["strategy"]
        all_nodes: list[ChunkNode] = []

        for doc_chunk in document_chunks:

            if doc_chunk.modality in {"table", "image_caption"}:
                all_nodes.append(ChunkNode(chunk=doc_chunk, level="paragraph"))
                continue

            if strategy == "fixed":
                nodes = self._fixed_chunking(doc_chunk)
            elif strategy == "semantic":
                nodes = self._semantic_chunking(doc_chunk)
            elif strategy == "hierarchical":
                nodes = self._hierarchical_chunking(doc_chunk)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy!r}")

            all_nodes.extend(nodes)

        logger.info(
            f"Chunking produced {len(all_nodes)} nodes "
            f"from {len(document_chunks)} consolidated sections "
            f"[strategy={strategy}]"
        )
        return all_nodes

    def _fixed_chunking(self, source: ParsedChunk) -> list[ChunkNode]:
        """
        Naive token-window chunking.  Breaks sentences arbitrarily.
        Included only as a baseline for ablation studies.
        """
        chunk_size = self._chunk_cfg["chunk_size"]
        overlap = self._chunk_cfg["chunk_overlap"]
        tokens = self._tokeniser.encode(source.text)
        nodes: list[ChunkNode] = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            window_tokens = tokens[start:end]
            text = self._tokeniser.decode(window_tokens)

            if len(window_tokens) >= self._chunk_cfg["min_chunk_size"] or end == len(tokens):
                nodes.append(
                    ChunkNode(
                        chunk=self._clone_chunk(source, text),
                        level="paragraph",
                    )
                )
            start += chunk_size - overlap

        return nodes

    def _semantic_chunking(
        self,
        source: ParsedChunk,
    ) -> list[ChunkNode]:
        """
        Split text into semantic chunks using embedding similarity.
        """
        chunk_size = self._chunk_cfg["override_chunk_size"] or self._chunk_cfg["chunk_size"]
        min_size = self._chunk_cfg["min_chunk_size"]
        percentile = self._chunk_cfg.get("breakpoint_percentile", 0.25)
        window_size = self._chunk_cfg.get("window_size", 3)

        sentences = self._split_sentences(source.text)

        if len(sentences) <= window_size:
            text = source.text.strip()
            if len(self._tokeniser.encode(text)) < min_size:
                return []
            return [
                ChunkNode(
                    chunk=self._clone_chunk(source, text),
                    level="paragraph",
                )
            ]

        windows = self._build_windows(sentences, window_size)
        embed = self._get_embedder()
        embeddings = embed(windows)

        similarities = self._adjacent_similarities(embeddings)

        breakpoint_indices = self._detect_breakpoints(similarities, percentile)

        segments = self._sentences_to_segments(sentences, breakpoint_indices)

        segments = self._merge_small_segments(segments, min_size)
        segments = self._split_large_segments(segments, chunk_size, source)

        nodes: list[ChunkNode] = []
        for seg_text in segments:
            tok_count = len(self._tokeniser.encode(seg_text))
            if tok_count < min_size and nodes:
                prev_text = nodes[-1].chunk.text + " " + seg_text
                nodes[-1] = ChunkNode(
                    chunk=self._clone_chunk(source, prev_text.strip()),
                    level="paragraph",
                )
            else:
                nodes.append(
                    ChunkNode(
                        chunk=self._clone_chunk(source, seg_text),
                        level="paragraph",
                    )
                )

        if not nodes:
            nodes.append(
                ChunkNode(
                    chunk=self._clone_chunk(source, source.text.strip()),
                    level="paragraph",
                )
            )

        return nodes

    def _hierarchical_chunking(self, source: ParsedChunk) -> list[ChunkNode]:
        """
        Build document, section, and paragraph-level chunks.
        """
        doc_id = str(uuid.uuid4())
        document_node = ChunkNode(
            chunk=self._clone_chunk(source, source.text),
            level="document",
            parent_id=None,
            children_ids=[],
        )
        document_node.chunk.chunk_id = doc_id

        section_nodes: list[ChunkNode] = []
        paragraph_nodes: list[ChunkNode] = []

        sections = self._split_sections(source.text)

        for section_text in sections:
            section_id = str(uuid.uuid4())

            heading = _extract_heading(section_text)
            depth = _heading_depth(section_text)

            section_clone = self._clone_chunk(source, section_text)
            section_clone.metadata["section"] = heading
            section_clone.metadata["section_depth"] = depth

            section_node = ChunkNode(
                chunk=section_clone,
                level="section",
                parent_id=doc_id,
                children_ids=[],
            )
            section_node.chunk.chunk_id = section_id
            document_node.children_ids.append(section_id)
            section_nodes.append(section_node)

            para_source = self._clone_chunk(source, section_text)
            para_source.metadata["section"] = heading
            para_nodes = self._semantic_chunking(para_source)

            for para_node in para_nodes:
                para_node.parent_id = section_id
                para_node.level = "paragraph"
                para_node.chunk.metadata["section"] = heading
                para_node.chunk.metadata["section_depth"] = depth
                para_node.chunk.metadata["parent_section_id"] = section_id
                section_node.children_ids.append(para_node.chunk.chunk_id or "")
                paragraph_nodes.append(para_node)

        return [document_node] + section_nodes + paragraph_nodes

    @staticmethod
    def _build_windows(sentences: list[str], window_size: int) -> list[str]:
        windows = []
        n = len(sentences)
        for i in range(n):
            end = min(i + window_size, n)
            window = " ".join(sentences[i:end])
            windows.append(window)
        return windows

    @staticmethod
    def _adjacent_similarities(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity between adjacent embedding windows.
        """
        n = len(embeddings)
        sims = np.zeros(n - 1, dtype=np.float32)
        for i in range(n - 1):
            a = embeddings[i].reshape(1, -1)
            b = embeddings[i + 1].reshape(1, -1)
            sims[i] = cosine_similarity(a, b)[0, 0]
        return sims

    @staticmethod
    def _detect_breakpoints(
        similarities: np.ndarray,
        percentile: float,
    ) -> list[int]:
        """
        Identify semantic split positions from similarity scores.
        """
        if len(similarities) == 0:
            return []

        threshold = float(np.quantile(similarities, percentile))

        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i)

        return breakpoints

    @staticmethod
    def _sentences_to_segments(
        sentences: list[str],
        breakpoint_indices: list[int],
    ) -> list[str]:
        """
        Group sentences into segments from breakpoint indices.
        """
        if not breakpoint_indices:
            return [" ".join(sentences)]

        segments: list[str] = []
        start = 0
        for bp in sorted(set(breakpoint_indices)):
            end = bp + 1
            segment = " ".join(sentences[start:end]).strip()
            if segment:
                segments.append(segment)
            start = end

        tail = " ".join(sentences[start:]).strip()
        if tail:
            segments.append(tail)

        return segments

    def _merge_small_segments(
        self,
        segments: list[str],
        min_size: int,
    ) -> list[str]:
        """
        Merge undersized segments into neighboring chunks.
        """
        if not segments:
            return segments

        merged: list[str] = []
        i = 0
        while i < len(segments):
            tok_count = len(self._tokeniser.encode(segments[i]))
            if tok_count < min_size:
                if i + 1 < len(segments):
                    segments[i + 1] = segments[i] + " " + segments[i + 1]
                    i += 1
                    continue
                elif merged:
                    merged[-1] = merged[-1] + " " + segments[i]
                    i += 1
                    continue
            merged.append(segments[i])
            i += 1

        return merged

    def _split_large_segments(
        self,
        segments: list[str],
        chunk_size: int,
        source: ParsedChunk,
    ) -> list[str]:
        """
        Split oversized segments while preserving sentence boundaries.
        """
        result: list[str] = []
        for segment in segments:
            tok_count = len(self._tokeniser.encode(segment))
            if tok_count <= chunk_size:
                result.append(segment)
                continue

            sentences = self._split_sentences(segment)
            current_sents: list[str] = []
            current_count = 0

            for sent in sentences:
                sent_toks = len(self._tokeniser.encode(sent))
                if current_count + sent_toks > chunk_size and current_sents:
                    result.append(" ".join(current_sents).strip())
                    current_sents = []
                    current_count = 0
                current_sents.append(sent)
                current_count += sent_toks

            if current_sents:
                result.append(" ".join(current_sents).strip())

        return result

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into sentences while preserving Markdown tables.
        """
        parts = re.split(r"(\n(?:\|[^\n]*\n)+)", "\n" + text)

        result: list[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if stripped.startswith("|"):
                result.append(stripped)
            else:
                result.extend(_split_on_boundaries(stripped))

        return result

    @staticmethod
    def _split_sections(text: str) -> list[str]:
        """
        Split Markdown text into heading-based sections.
        """
        sections = re.split(r"(?m)(?=^#{1,3}\s+)", text)

        if len(sections) > 1:
            return [s.strip() for s in sections if s.strip()]

        sections = text.split("\n\n")
        return [s.strip() for s in sections if s.strip()]

    @staticmethod
    def _clone_chunk(source: ParsedChunk, new_text: str) -> ParsedChunk:
        """
        Clone a chunk with updated text.
        """
        clone = ParsedChunk(
            text=new_text,
            metadata=dict(source.metadata),
            source_file=source.source_file,
            source_name=source.source_name,
            modality=source.modality,
            language=source.language,
            doc_version=source.doc_version,
            ingestion_ts=source.ingestion_ts,
        )
        clone.chunk_id = clone.compute_fingerprint()
        return clone


def _extract_heading(section_text: str) -> str:
    match = _HEADING_RE.search(section_text)
    return match.group(2).strip() if match else ""


def _heading_depth(section_text: str) -> int:
    match = _HEADING_RE.search(section_text)
    return len(match.group(1)) if match else 0
