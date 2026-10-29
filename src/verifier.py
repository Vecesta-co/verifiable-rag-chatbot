"""Passage verification, snippet hash checking, and groundedness scoring."""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import hashlib
from typing import List, Dict
from .models import DocumentChunk, Citation, EvalReport
from .index import tokenize

def verify_citation(citation: Citation, chunk: DocumentChunk) -> bool:
    expected_hash = hashlib.sha256(chunk.content.strip().encode("utf-8")).hexdigest()[:12]
    return citation.content_hash == expected_hash

def compute_groundedness(answer: str, citations: List[Citation]) -> float:
    if not citations or not answer.strip():
        return 0.0
    ans_tokens = set(tokenize(answer))
    if not ans_tokens:
        return 0.0
        
    citation_tokens = set()
    for c in citations:
        citation_tokens.update(tokenize(c.snippet))
        
    overlap = ans_tokens.intersection(citation_tokens)
    return round(len(overlap) / len(ans_tokens), 4)

def evaluate_rag(query: str, answer: str, citations: List[Citation], chunk_map: Dict[str, DocumentChunk]) -> EvalReport:
    verified_count = 0
    for c in citations:
        chk = chunk_map.get(c.chunk_id)
        if chk and verify_citation(c, chk):
            verified_count += 1
            
    groundedness = compute_groundedness(answer, citations)
    coverage = round(verified_count / max(1, len(citations)), 4)
    
    return EvalReport(
        query=query,
        total_citations=len(citations),
        verified_citations=verified_count,
        groundedness_score=groundedness,
        citation_coverage=coverage
    )
