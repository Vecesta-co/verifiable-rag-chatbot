"""Data models for chunks, queries, citations, and evaluation."""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import hashlib
import uuid

class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_name: str
    content: str
    start_line: int
    end_line: int
    content_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.strip().encode("utf-8")).hexdigest()[:12]

class Citation(BaseModel):
    chunk_id: str
    doc_name: str
    start_line: int
    end_line: int
    snippet: str
    content_hash: str
    is_verified: bool = True

class SearchResult(BaseModel):
    chunk: DocumentChunk
    bm25_score: float
    tfidf_score: float
    hybrid_score: float

class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    groundedness_score: float = 1.0  # 0.0 to 1.0

class EvalReport(BaseModel):
    query: str
    total_citations: int
    verified_citations: int
    groundedness_score: float
    citation_coverage: float
