"""Comprehensive test suite for verifiable-rag-chatbot (25 tests)."""
import pytest
import os
import tempfile
from src.models import DocumentChunk, Citation, SearchResult, RAGResponse, EvalReport
from src.chunker import chunk_by_paragraphs, chunk_fixed_size
from src.index import tokenize, HybridIndex
from src.verifier import verify_citation, compute_groundedness, evaluate_rag
from src.agent import VerifiableRAGAgent

SAMPLE_TEXT = """Quantum computing leverages quantum mechanical phenomena.
Superposition and entanglement are key principles.

These principles allow quantum computers to solve certain problems exponentially faster than classical computers.
Applications include cryptography and molecular modeling.

However, fault-tolerant error correction remains a significant hardware challenge today.
Researchers worldwide are exploring trapped ions and superconducting qubits.
"""

# 1. Models & Chunking Tests
def test_document_chunk_hash():
    chk = DocumentChunk(doc_name="test.txt", content="Hello world", start_line=1, end_line=1)
    assert len(chk.content_hash) == 12
    assert chk.content_hash == DocumentChunk(doc_name="other.txt", content="Hello world", start_line=5, end_line=5).content_hash

def test_chunk_by_paragraphs_count():
    chunks = chunk_by_paragraphs("quantum.txt", SAMPLE_TEXT)
    assert len(chunks) == 3

def test_chunk_by_paragraphs_line_numbers():
    chunks = chunk_by_paragraphs("quantum.txt", SAMPLE_TEXT)
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2
    assert chunks[1].start_line == 4
    assert chunks[1].end_line == 5
    assert chunks[2].start_line == 7
    assert chunks[2].end_line == 8

def test_chunk_fixed_size():
    chunks = chunk_fixed_size("quantum.txt", SAMPLE_TEXT, chunk_lines=3, overlap=1)
    assert len(chunks) >= 3
    assert chunks[0].start_line == 1

def test_chunk_empty_text():
    assert chunk_by_paragraphs("empty.txt", "") == []
    assert chunk_fixed_size("empty.txt", "") == []

# 2. Tokenizer & Index Tests
def test_tokenize_clean():
    tokens = tokenize("Superposition and entanglement are KEY principles!")
    assert "superposition" in tokens
    assert "entanglement" in tokens
    assert "principles" in tokens
    assert "and" not in tokens
    assert "are" not in tokens

def test_hybrid_index_empty_search():
    index = HybridIndex()
    assert index.search("quantum") == []

def test_hybrid_index_search():
    index = HybridIndex()
    chunks = chunk_by_paragraphs("quantum.txt", SAMPLE_TEXT)
    index.add_chunks(chunks)
    
    results = index.search("superposition entanglement", top_k=2)
    assert len(results) > 0
    assert "superposition" in results[0].chunk.content.lower()

def test_hybrid_index_bm25_vs_tfidf_scores():
    index = HybridIndex()
    chunks = chunk_by_paragraphs("quantum.txt", SAMPLE_TEXT)
    index.add_chunks(chunks)
    
    results = index.search("cryptography molecular", top_k=1)
    assert len(results) == 1
    assert results[0].bm25_score > 0
    assert results[0].tfidf_score > 0
    assert results[0].hybrid_score > 0

def test_hybrid_index_no_match():
    index = HybridIndex()
    chunks = chunk_by_paragraphs("quantum.txt", SAMPLE_TEXT)
    index.add_chunks(chunks)
    
    results = index.search("completely unrelated dinosaurs fossils")
    assert len(results) == 0

# 3. Verifier & Groundedness Tests
def test_verify_citation_valid():
    chk = DocumentChunk(doc_name="test.txt", content="Specific proof passage.", start_line=1, end_line=1)
    cit = Citation(
        chunk_id=chk.id,
        doc_name=chk.doc_name,
        start_line=chk.start_line,
        end_line=chk.end_line,
        snippet=chk.content,
        content_hash=chk.content_hash
    )
    assert verify_citation(cit, chk) is True

def test_verify_citation_tampered_hash():
    chk = DocumentChunk(doc_name="test.txt", content="Specific proof passage.", start_line=1, end_line=1)
    cit = Citation(
        chunk_id=chk.id,
        doc_name=chk.doc_name,
        start_line=chk.start_line,
        end_line=chk.end_line,
        snippet=chk.content,
        content_hash="tampered1234"
    )
    assert verify_citation(cit, chk) is False

def test_compute_groundedness_perfect_overlap():
    cit = Citation(
        chunk_id="1", doc_name="doc.txt", start_line=1, end_line=1,
        snippet="Quantum computers solve cryptography problems rapidly.", content_hash="hash"
    )
    score = compute_groundedness("Quantum computers solve cryptography problems", [cit])
    assert score == 1.0

def test_compute_groundedness_zero_overlap():
    cit = Citation(
        chunk_id="1", doc_name="doc.txt", start_line=1, end_line=1,
        snippet="Apples and oranges are fruits.", content_hash="hash"
    )
    score = compute_groundedness("Quantum computers solve cryptography problems", [cit])
    assert score == 0.0

def test_compute_groundedness_empty_citations():
    score = compute_groundedness("Some answer", [])
    assert score == 0.0

def test_evaluate_rag_report():
    chk = DocumentChunk(doc_name="test.txt", content="Trapped ions superconducting qubits.", start_line=1, end_line=1)
    cit = Citation(
        chunk_id=chk.id, doc_name=chk.doc_name, start_line=1, end_line=1,
        snippet=chk.content, content_hash=chk.content_hash
    )
    rep = evaluate_rag("qubits", "Trapped ions superconducting qubits", [cit], {chk.id: chk})
    assert rep.total_citations == 1
    assert rep.verified_citations == 1
    assert rep.citation_coverage == 1.0
    assert rep.groundedness_score == 1.0

# 4. Agent Tests
def test_agent_ingest_text():
    agent = VerifiableRAGAgent()
    count = agent.ingest_text("doc.txt", SAMPLE_TEXT)
    assert count == 3
    assert len(agent.chunk_map) == 3

def test_agent_ingest_file(tmp_path):
    f = tmp_path / "sample.md"
    f.write_text(SAMPLE_TEXT, encoding="utf-8")
    
    agent = VerifiableRAGAgent()
    count = agent.ingest_file(str(f))
    assert count == 3

def test_agent_answer_query_found():
    agent = VerifiableRAGAgent()
    agent.ingest_text("quantum.txt", SAMPLE_TEXT)
    
    resp = agent.answer_query("superconducting qubits hardware challenge")
    assert resp.groundedness_score > 0.0
    assert len(resp.citations) > 0
    assert "quantum.txt" in resp.citations[0].doc_name
    assert resp.citations[0].start_line == 7
    assert resp.citations[0].end_line == 8

def test_agent_answer_query_not_found():
    agent = VerifiableRAGAgent()
    agent.ingest_text("quantum.txt", SAMPLE_TEXT)
    
    resp = agent.answer_query("space travel mars colonization")
    assert resp.groundedness_score == 0.0
    assert len(resp.citations) == 0

def test_agent_multiple_documents():
    agent = VerifiableRAGAgent()
    agent.ingest_text("doc1.txt", "Photosynthesis converts light into chemical energy.")
    agent.ingest_text("doc2.txt", "Cellular respiration produces ATP in mitochondria.")
    
    resp1 = agent.answer_query("photosynthesis light energy")
    assert any(c.doc_name == "doc1.txt" for c in resp1.citations)
    
    resp2 = agent.answer_query("mitochondria ATP respiration")
    assert any(c.doc_name == "doc2.txt" for c in resp2.citations)

def test_citation_line_range_format():
    agent = VerifiableRAGAgent()
    agent.ingest_text("doc.txt", "Alpha bravo\nCharlie delta\n\nEcho foxtrot\nGolf hotel")
    resp = agent.answer_query("Echo foxtrot")
    assert len(resp.citations) > 0
    assert resp.citations[0].start_line == 4
    assert resp.citations[0].end_line == 5
