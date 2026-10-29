"""REST API server for verifiable-rag-chatbot."""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from .agent import VerifiableRAGAgent
from .verifier import evaluate_rag

AGENT = VerifiableRAGAgent()

class RAGServerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json({"status": "ok", "indexed_chunks": len(AGENT.chunk_map)})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if parsed.path == "/ingest":
            doc_name = body.get("doc_name", "document.txt")
            text = body.get("text", "")
            strategy = body.get("strategy", "paragraph")
            count = AGENT.ingest_text(doc_name, text, strategy)
            self._json({"status": "ingested", "doc_name": doc_name, "chunks_created": count}, 201)

        elif parsed.path == "/chat":
            query = body.get("query", "")
            top_k = body.get("top_k", 3)
            resp = AGENT.answer_query(query, top_k=top_k)
            self._json(resp.model_dump())

        elif parsed.path == "/eval":
            query = body.get("query", "")
            top_k = body.get("top_k", 3)
            resp = AGENT.answer_query(query, top_k=top_k)
            report = evaluate_rag(query, resp.answer, resp.citations, AGENT.chunk_map)
            self._json(report.model_dump())
        else:
            self._json({"error": "not found"}, 404)

def run_server(host="0.0.0.0", port=8080):
    server = HTTPServer((host, port), RAGServerHandler)
    print(f"Verifiable RAG server running on http://{host}:{port}")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
