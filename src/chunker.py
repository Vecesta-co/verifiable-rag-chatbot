"""Document ingestion and chunking with line number tracking."""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
from typing import List
from .models import DocumentChunk

def chunk_by_paragraphs(doc_name: str, text: str) -> List[DocumentChunk]:
    lines = text.splitlines()
    chunks: List[DocumentChunk] = []
    
    current_lines: List[str] = []
    start_line = 1
    
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append(DocumentChunk(
                        doc_name=doc_name,
                        content=chunk_text,
                        start_line=start_line,
                        end_line=i - 1
                    ))
                current_lines = []
            start_line = i + 1
        else:
            if not current_lines:
                start_line = i
            current_lines.append(line)
            
    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append(DocumentChunk(
                doc_name=doc_name,
                content=chunk_text,
                start_line=start_line,
                end_line=len(lines)
            ))
            
    return chunks

def chunk_fixed_size(doc_name: str, text: str, chunk_lines: int = 5, overlap: int = 1) -> List[DocumentChunk]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks: List[DocumentChunk] = []
    step = max(1, chunk_lines - overlap)
    
    for i in range(0, len(lines), step):
        slice_lines = lines[i:i + chunk_lines]
        chunk_text = "\n".join(slice_lines).strip()
        if chunk_text:
            chunks.append(DocumentChunk(
                doc_name=doc_name,
                content=chunk_text,
                start_line=i + 1,
                end_line=min(len(lines), i + len(slice_lines))
            ))
        if i + chunk_lines >= len(lines):
            break
            
    return chunks
