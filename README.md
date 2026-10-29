# verifiable-rag-chatbot

> **Grounded retrieval-augmented generation engine with verifiable exact passage citations, hybrid BM25 + TF-IDF retrieval, and cryptographic SHA256 integrity verification.**

[![CI](https://github.com/Vecesta-co/verifiable-rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Vecesta-co/verifiable-rag-chatbot/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)

```
$ python -m src.cli ingest sample_doc.md
[OK] Ingested sample_doc.md -> created 5 chunks.

$ python -m src.cli chat "quantum hardware challenge"
Query: quantum hardware challenge
Answer:
According to sample_doc.md (lines 7-8): Fault-tolerant error correction remains a significant hardware challenge today. [1]

Citations:
  [1] sample_doc.md (lines 7-8) [SHA256:4a91f53e201b]

Groundedness Score: 100.0%

$ python -m src.cli eval "quantum hardware challenge"
=== Verifiable RAG Evaluation ===
Query:               quantum hardware challenge
Total Citations:     1
Verified Citations:  1
Citation Coverage:   100.0%
Groundedness Score:  100.0%
```

## Architecture

```
┌──────────────────┐
│ Ingest Document  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Line-Aware Chunking (Paragraph / Fixed)     │
│  - Start Line, End Line, SHA256 Content Hash │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Hybrid Inverted Index                       │
│  - BM25 Okapi (Keywords)                     │
│  - TF-IDF Cosine (Semantic Frequency)        │
└──────────────────────┬───────────────────────┘
                       │
       Query ──────────►
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Verification & Evaluation Engine            │
│  - Citation Verification (Hash Match)        │
│  - Groundedness Ratio (Lexical Recall)       │
└──────────────────────────────────────────────┘
```

## Comparison

| Metric / Capability | verifiable-rag-chatbot | Typical Naive RAG |
|---|---|---|
| Line-Level Citations | Yes (Exact start/end lines) | No (Vague chunk / page) |
| Tamper-Proof Citations | Yes (SHA256 truncated hash) | None |
| Groundedness Evaluation | Built-in (RAGAS-style metrics) | Requires external paid API |
| Hybrid Search | Native BM25 + TF-IDF (Zero external deps) | External Vector DB required |
| Local Execution | 100% Offline & Pure Python | Cloud dependency |

## Key Features

- **Exact Passage Tracking**: Every chunk retains exact file name and line boundaries (`start_line`, `end_line`).
- **Cryptographic Provenance**: Chunks and citations are paired with SHA-256 fingerprint hashes to prevent hallucinated quotes.
- **Pure-Python Hybrid Retrieval**: BM25 Okapi + TF-IDF vector blending with configurable `alpha` weights.
- **REST & CLI Interfaces**: Complete HTTP endpoints (`/ingest`, `/chat`, `/eval`) and terminal CLI.

## Quick Start

```bash
git clone https://github.com/Vecesta-co/verifiable-rag-chatbot.git
cd verifiable-rag-chatbot
pip install pydantic pytest

# Ask a query against a document:
python -m src.cli chat "what are the main challenges" --file my_doc.txt

# Start REST server:
python -m src.server
```

## Difficulties & Solutions

- **Exact Line Accounting**: Document chunking frequently drops line numbering when stripping whitespace. We built a boundary-preserving paragraph chunker that computes exact line offsets before tokenization.
- **Hallucination Prevention**: Integrated a token-overlap groundedness scoring algorithm that flags when answers introduce terms unsupported by citations.

---

**Built by Vecesta.co**
- Email: vecesta.co@gmail.com
- LinkedIn: https://www.linkedin.com/in/vansh-dhawan-2625v
- GitHub: https://github.com/Vecesta-co
- X: https://x.com/Vansh_Dhawann
- Hugging Face: https://huggingface.co/Vecesta
