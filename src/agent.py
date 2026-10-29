"""Verifiable RAG Agent combining hybrid retrieval and citation generation."""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
from typing import List, Dict, Optional
from .models import DocumentChunk, Citation, RAGResponse
from .chunker import chunk_by_paragraphs, chunk_fixed_size
from .index import HybridIndex
from .verifier import compute_groundedness

class VerifiableRAGAgent:
    def __init__(self, index: Optional[HybridIndex] = None):
        self.index = index or HybridIndex()
        self.chunk_map: Dict[str, DocumentChunk] = {}
        
    def ingest_text(self, doc_name: str, text: str, strategy: str = "paragraph") -> int:
        if strategy == "fixed":
            chunks = chunk_fixed_size(doc_name, text)
        else:
            chunks = chunk_by_paragraphs(doc_name, text)
            
        for c in chunks:
            self.chunk_map[c.id] = c
            
        self.index.add_chunks(chunks)
        return len(chunks)

    def ingest_file(self, file_path: str, strategy: str = "paragraph") -> int:
        doc_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return self.ingest_text(doc_name, text, strategy)

    def answer_query(self, query: str, top_k: int = 3) -> RAGResponse:
        results = self.index.search(query, top_k=top_k)
        if not results:
            return RAGResponse(
                query=query,
                answer="No relevant documents found to answer your query.",
                citations=[],
                groundedness_score=0.0
            )
            
        citations: List[Citation] = []
        answer_parts: List[str] = []
        
        for idx, res in enumerate(results, start=1):
            chk = res.chunk
            snippet = chk.content[:160].replace("\n", " ")
            citation = Citation(
                chunk_id=chk.id,
                doc_name=chk.doc_name,
                start_line=chk.start_line,
                end_line=chk.end_line,
                snippet=chk.content,
                content_hash=chk.content_hash,
                is_verified=True
            )
            citations.append(citation)
            answer_parts.append(f"According to {chk.doc_name} (lines {chk.start_line}-{chk.end_line}): {snippet} [{idx}]")
            
        synthesized_answer = " ".join(answer_parts)
        groundedness = compute_groundedness(synthesized_answer, citations)
        
        return RAGResponse(
            query=query,
            answer=synthesized_answer,
            citations=citations,
            groundedness_score=groundedness
        )
