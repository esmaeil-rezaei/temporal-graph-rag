from __future__ import annotations

from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    number: int | None = None
    chunk_id: str | None = None
    source_file: str | None = None
    source_name: str | None = None
    ingestion_ts: str | None = None
    excerpt: str | None = None


class GenerationOutput(BaseModel):

    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    faithfulness_score: float | None = None
    has_conflict: bool = False


class DirectResponseOutput(BaseModel):
    answer: str
    intent: str
