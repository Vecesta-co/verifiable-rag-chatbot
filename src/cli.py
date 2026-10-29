"""CLI for verifiable-rag-chatbot."""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import os
from .agent import VerifiableRAGAgent
from .verifier import evaluate_rag

GLOBAL_AGENT = VerifiableRAGAgent()

def cmd_ingest(args):
    path = args.path
    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist.")
        return
    count = GLOBAL_AGENT.ingest_file(path, strategy=args.strategy)
    print(f"[OK] Ingested {path} -> created {count} chunks.")

def cmd_chat(args):
    if not GLOBAL_AGENT.chunk_map and args.file:
        GLOBAL_AGENT.ingest_file(args.file)
    resp = GLOBAL_AGENT.answer_query(args.query, top_k=args.top_k)
    print(f"\nQuery: {resp.query}")
    print(f"Answer:\n{resp.answer}\n")
    print("Citations:")
    for idx, c in enumerate(resp.citations, start=1):
        print(f"  [{idx}] {c.doc_name} (lines {c.start_line}-{c.end_line}) [SHA256:{c.content_hash}]")
    print(f"\nGroundedness Score: {resp.groundedness_score * 100:.1f}%\n")

def cmd_eval(args):
    if not GLOBAL_AGENT.chunk_map and args.file:
        GLOBAL_AGENT.ingest_file(args.file)
    resp = GLOBAL_AGENT.answer_query(args.query)
    report = evaluate_rag(resp.query, resp.answer, resp.citations, GLOBAL_AGENT.chunk_map)
    print("\n=== Verifiable RAG Evaluation ===")
    print(f"Query:               {report.query}")
    print(f"Total Citations:     {report.total_citations}")
    print(f"Verified Citations:  {report.verified_citations}")
    print(f"Citation Coverage:   {report.citation_coverage * 100:.1f}%")
    print(f"Groundedness Score:  {report.groundedness_score * 100:.1f}%\n")

def main():
    parser = argparse.ArgumentParser(prog="rag", description="Verifiable RAG Chatbot CLI")
    sub = parser.add_subparsers(dest="command")

    p_ing = sub.add_parser("ingest", help="Ingest a document file")
    p_ing.add_argument("path", help="Path to text or markdown file")
    p_ing.add_argument("--strategy", default="paragraph", choices=["paragraph", "fixed"])

    p_chat = sub.add_parser("chat", help="Ask a question against ingested documents")
    p_chat.add_argument("query", help="Question to ask")
    p_chat.add_argument("--file", default="", help="Optional document file to ingest first")
    p_chat.add_argument("--top-k", type=int, default=3, dest="top_k")

    p_eval = sub.add_parser("eval", help="Evaluate groundedness and citation verification")
    p_eval.add_argument("query", help="Evaluation query")
    p_eval.add_argument("--file", default="", help="Document file to evaluate against")

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
