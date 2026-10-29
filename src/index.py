"""Hybrid search index: BM25 keyword matching + TF-IDF cosine similarity."""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import math
import re
from typing import List, Dict, Tuple
from .models import DocumentChunk, SearchResult

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","i","my",
    "you","your","he","she","it","they","their","this","that","these","those"
}

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

class HybridIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {} # term -> [(chunk_idx, count)]
        self.df: Dict[str, int] = {}
        
    def add_chunks(self, chunks: List[DocumentChunk]):
        self.chunks.extend(chunks)
        self._rebuild()
        
    def _rebuild(self):
        self.inverted_index.clear()
        self.df.clear()
        self.doc_lens = []
        
        total_len = 0
        for idx, chunk in enumerate(self.chunks):
            tokens = tokenize(chunk.content)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
                
            for t, cnt in counts.items():
                if t not in self.inverted_index:
                    self.inverted_index[t] = []
                self.inverted_index[t].append((idx, cnt))
                self.df[t] = self.df.get(t, 0) + 1
                
        self.avg_doc_len = total_len / max(1, len(self.chunks))

    def _bm25_score(self, query_tokens: List[str], chunk_idx: int) -> float:
        N = len(self.chunks)
        score = 0.0
        doc_len = self.doc_lens[chunk_idx]
        
        for q in query_tokens:
            if q not in self.df:
                continue
            df = self.df[q]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            # find tf in this doc
            tf = 0
            for c_idx, count in self.inverted_index[q]:
                if c_idx == chunk_idx:
                    tf = count
                    break
            if tf > 0:
                denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1e-6, self.avg_doc_len)))
                score += idf * ((tf * (self.k1 + 1)) / denom)
        return score

    def _tfidf_score(self, query_tokens: List[str], chunk_idx: int) -> float:
        N = len(self.chunks)
        score = 0.0
        doc_len = max(1, self.doc_lens[chunk_idx])
        
        for q in query_tokens:
            if q not in self.df:
                continue
            df = self.df[q]
            idf = math.log((N + 1.0) / (df + 1.0)) + 1.0
            
            tf = 0
            for c_idx, count in self.inverted_index[q]:
                if c_idx == chunk_idx:
                    tf = count
                    break
            if tf > 0:
                tf_norm = tf / doc_len
                score += tf_norm * idf
        return score

    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[SearchResult]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return []
            
        # Candidate chunks that match at least one token
        candidate_indices = set()
        for q in q_tokens:
            if q in self.inverted_index:
                for c_idx, _ in self.inverted_index[q]:
                    candidate_indices.add(c_idx)
                    
        results: List[SearchResult] = []
        for idx in candidate_indices:
            bm25 = self._bm25_score(q_tokens, idx)
            tfidf = self._tfidf_score(q_tokens, idx)
            hybrid = alpha * bm25 + (1 - alpha) * tfidf
            results.append(SearchResult(
                chunk=self.chunks[idx],
                bm25_score=round(bm25, 4),
                tfidf_score=round(tfidf, 4),
                hybrid_score=round(hybrid, 4)
            ))
            
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results[:top_k]
